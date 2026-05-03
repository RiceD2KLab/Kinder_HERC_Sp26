"""Configuration objects for the research-chunk classification pipeline.

This module centralizes every parameter that controls the pipeline:
  - dataset loading
  - transcript-level splitting
  - embedding generation
  - feature construction
  - model training
  - validation-time threshold selection
  - output artifact locations

Design decisions baked into the defaults
-----------------------------------------
Split (60 / 20 / 20 at the transcript level)
    With 15 transcripts this yields 9 train / 3 val / 3 test.
    Narrower val/test (e.g. 70/15/15) would leave only 2 val transcripts,
    making threshold tuning unreliable.  When the corpus grows to 20
    transcripts the same fractions give 12/4/4 with no config change needed.

Embeddings (all-mpnet-base-v2)
    Strong general-purpose sentence model; 768-d embeddings; runs on CPU
    in reasonable time for ~15 transcripts.

Feature mode (chunk_only)
    Each example = embed(chunk).  The optional "query_conditioned" mode
    appends a fixed query embedding, doubling the feature dimension.

Threshold selection
    The threshold that maximises the F2 score on the validation set is
    selected.  F2 weights recall twice as heavily as precision, favouring
    models that miss fewer research mentions.  The same F2-first ordering
    is used to compare models across the C x class_weight grid.

Class weighting (swept: None vs "balanced")
    "balanced" re-weights the logistic loss so missing a positive is
    penalised more than missing a negative.  We sweep both options and let
    validation choose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# Type alias for the feature-mode option.
FeatureMode = Literal["chunk_only", "query_conditioned"]

# Type alias for model type.
ModelType = Literal["logistic_regression", "xgboost"]


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SplitConfig:
    """Parameters that control transcript-level train / val / test splitting.

    Splitting at the transcript level (one whole meeting per split) is
    mandatory: chunks from the same meeting are highly correlated, so
    mixing them across splits would leak information and inflate metrics.

    Attributes:
        train_fraction: Fraction of transcripts assigned to training.
        val_fraction:   Fraction of transcripts assigned to validation.
        test_fraction:  Fraction of transcripts assigned to testing.
        random_seed:    Seed used for deterministic tie-breaking during
                        the greedy assignment step.
    """

    train_fraction: float = 0.6
    val_fraction:   float = 0.2
    test_fraction:  float = 0.2
    random_seed:    int   = 42
    stratify:       bool  = True

    def validate(self) -> None:
        """Raise ValueError if the fractions do not sum to 1.0.

        Inputs:
            None — reads self.train_fraction, self.val_fraction, self.test_fraction.

        Outputs:
            None — raises on invalid configuration.

        Raises:
            ValueError: If the three fractions do not sum to 1.0.
        """
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Split fractions must sum to 1.0. "
                f"Got {self.train_fraction} + {self.val_fraction} + "
                f"{self.test_fraction} = {total}."
            )


@dataclass(slots=True)
class EmbeddingConfig:
    """Parameters for text embedding and feature-matrix construction.

    Attributes:
        model_name:           SentenceTransformer model identifier.
        batch_size:           Rows encoded per forward pass.
        normalize_embeddings: Whether to L2-normalise each embedding vector.
        context_window:       Number of neighboring chunks on *each side* to
                              concatenate before embedding.  0 = chunk text only.
        text_column:          Column in the CSV that holds raw chunk text.
        feature_mode:         "chunk_only" uses the chunk embedding alone;
                              "query_conditioned" appends a fixed query embedding,
                              doubling the feature dimension.
        query_text:           Guiding question used when feature_mode is
                              "query_conditioned".
    """

    model_name:           str        = "sentence-transformers/all-mpnet-base-v2"
    batch_size:           int        = 32
    normalize_embeddings: bool       = True
    context_window:       int        = 0
    text_column:          str        = "text"
    feature_mode:         FeatureMode = "chunk_only"
    query_text:           str        = (
        "How are research, data, reports, or studies used to make informed decisions?"
    )


@dataclass(slots=True)
class ModelConfig:
    """Hyperparameter search space for the logistic regression baseline.

    Attributes:
        c_values:              Candidate inverse-regularisation strengths.
                               Smaller C = stronger regularisation.
        class_weight_options:  None keeps default equal weighting;
                               "balanced" up-weights positives proportionally
                               to the class imbalance — important here because
                               positive chunks are rare.
        max_iter:              Maximum solver iterations per fit call.
        random_seed:           Seed passed to LogisticRegression for
                               reproducible initialisation.
        solver:                Scikit-learn solver.  "liblinear" works well for
                               small-to-medium datasets with L1/L2 penalties.
        fixed_threshold:       If set, skip automatic threshold selection and
                               use this value directly.  Useful for manual
                               inspection runs.
    """

    c_values:              list[float]                         = field(
        default_factory=lambda: [0.25, 0.5, 1.0, 1.5, 2.0]
    )
    class_weight_options:  list[str | dict[int, float] | None] = field(
        default_factory=lambda: ["balanced"]
    )
    max_iter:              int         = 2_000
    random_seed:           int         = 42
    solver:                str         = "liblinear"
    fixed_threshold:       float | None = None
    full_train_eval:       bool        = False
    # LASSO feature-selection settings
    use_feature_selection: bool        = False
    lasso_c_values:        list[float] = field(
        default_factory=lambda: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    )
    lasso_criterion:       str         = "bic"

    # ------------------------------------------------------------------
    # Model type selection
    # ------------------------------------------------------------------
    model_type: ModelType = "logistic_regression"

    # ------------------------------------------------------------------
    # XGBoost hyperparameter search space
    # (only used when model_type = "xgboost")
    # ------------------------------------------------------------------
    xgb_n_estimators_options:  list[int]   = field(
        default_factory=lambda: [100, 300, 500]
    )
    xgb_max_depth_options:     list[int]   = field(
        default_factory=lambda: [3, 5, 7]
    )
    xgb_learning_rate_options: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.3]
    )
    xgb_device:                str         = "cpu"


@dataclass(slots=True)
class ThresholdConfig:
    """Controls how a probability threshold is chosen on the validation set.

    The threshold that maximises the F2 score on the validation set is
    selected.  F2 weights recall twice as heavily as precision, favouring
    models that miss fewer research mentions.

    Attributes:
        beta:                 Beta for F-beta scoring (2.0 = F2).
        candidate_thresholds: Grid of thresholds swept during validation.
    """

    beta:                 float       = 2.0
    candidate_thresholds: list[float] = field(
        default_factory=lambda: [round(t / 20, 2) for t in range(1, 20)]
        # [0.05, 0.10, ..., 0.95]
    )


@dataclass(slots=True)
class OutputConfig:
    """File paths for all pipeline output artifacts.

    Attributes:
        output_dir:                 Root directory for all outputs.
        save_embeddings:            If True, persist the raw feature matrix.
        embeddings_filename:        Filename for the optional .npy feature dump.
        predictions_filename:       Test-set predictions and probabilities.
        validation_sweep_filename:  Per-threshold val metrics for all model
                                    configurations.
        metrics_filename:           Compact JSON experiment summary.
        transcript_split_filename:  One row per transcript showing its split.
    """

    output_dir:                 Path = Path("outputs")
    save_embeddings:            bool = False
    embeddings_filename:        str  = "feature_matrix.npy"
    predictions_filename:       str  = "test_predictions.csv"
    validation_sweep_filename:  str  = "validation_threshold_sweep.csv"
    metrics_filename:           str  = "metrics_summary.json"
    transcript_split_filename:  str  = "transcript_split_assignments.csv"


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PipelineConfig:
    """Top-level configuration container for the full pipeline.

    Only ``transcript_data_dir`` is required at construction time; every
    other field has a sensible default.

    Attributes:
        transcript_data_dir: Directory that contains one CSV per transcript.
        split:               Split-fraction and seed settings.
        embedding:           Embedding model and feature-mode settings.
        model:               Logistic regression hyperparameter search space.
        threshold:           Validation-time threshold selection settings.
        output:              Output file locations.
    """

    transcript_data_dir: Path
    split:               SplitConfig     = field(default_factory=SplitConfig)
    embedding:           EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model:               ModelConfig     = field(default_factory=ModelConfig)
    threshold:           ThresholdConfig = field(default_factory=ThresholdConfig)
    output:              OutputConfig    = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """Check configuration consistency before starting a pipeline run.

        Inputs:
            None — reads all sub-config fields from self.

        Outputs:
            None — raises on invalid configuration.

        Raises:
            FileNotFoundError: If ``transcript_data_dir`` does not exist.
            ValueError:        If any sub-config values are inconsistent.
        """
        self.split.validate()

        if not self.transcript_data_dir.exists():
            raise FileNotFoundError(
                f"Transcript data directory does not exist: {self.transcript_data_dir}"
            )
        if self.embedding.feature_mode not in {"chunk_only", "query_conditioned"}:
            raise ValueError(
                "feature_mode must be 'chunk_only' or 'query_conditioned'."
            )
        if (
            self.embedding.feature_mode == "query_conditioned"
            and not self.embedding.query_text.strip()
        ):
            raise ValueError(
                "query_text must be non-empty when feature_mode='query_conditioned'."
            )
