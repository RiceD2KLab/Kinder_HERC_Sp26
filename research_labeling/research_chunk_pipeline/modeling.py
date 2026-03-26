"""Model training, threshold selection, and evaluation utilities.

Workflow
--------
1. ``train_logistic_regression``   - fit one model with given hyper-params
2. ``predict_positive_probabilities`` - get P(positive) for each example
3. ``sweep_thresholds``            - evaluate a grid of decision thresholds
4. ``select_threshold``            - pick the best threshold by F2 score
5. ``evaluate_predictions``        - compute all metrics at a chosen threshold
6. ``select_best_logistic_model``  - run the full grid search over C and class_weight

Threshold-selection philosophy
-------------------------------
The sponsor prefers *false positives over false negatives*: it is better to
flag extra chunks for review than to miss a genuine research mention.  The
pipeline selects the threshold that maximises the F2 score on the validation
set.  F2 weights recall twice as heavily as precision, favouring models that
miss fewer research mentions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Metrics container
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SplitMetrics:
    """All evaluation metrics for one split at one threshold.

    Attributes:
        threshold:               Decision threshold applied.
        precision:               TP / (TP + FP).
        recall:                  TP / (TP + FN).
        f1:                      Harmonic mean of precision and recall.
        f2:                      F-beta with beta=2 (recall-weighted).
        average_precision:       Area under the precision-recall curve.
        true_negative:           Count of correct negatives.
        false_positive:          Count of incorrectly flagged negatives.
        false_negative:          Count of missed positives (the costly error).
        true_positive:           Count of correctly flagged positives.
        predicted_positive_count: Total chunks flagged as positive.
        actual_positive_count:   Total true positive chunks in the split.
    """

    threshold:               float
    precision:               float
    recall:                  float
    f1:                      float
    f2:                      float
    average_precision:       float
    true_negative:           int
    false_positive:          int
    false_negative:          int
    true_positive:           int
    predicted_positive_count: int
    actual_positive_count:   int

    def to_dict(self) -> dict[str, float | int]:
        """Return a JSON-serialisable dict representation."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Model-selection result container
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ModelSelectionResult:
    """Everything produced by the validation-time grid search.

    Attributes:
        model:                 The winning fitted LogisticRegression.
        best_c:                C value of the winning model.
        best_class_weight:     class_weight of the winning model.
        best_threshold:        Threshold selected on the validation set.
        validation_metrics:    Metrics of the winning model at best_threshold.
        validation_sweep_rows: All threshold-sweep rows (for the CSV artifact).
    """

    model:                 LogisticRegression
    best_c:                float
    best_class_weight:     str | dict[int, float] | None
    best_threshold:        float
    validation_metrics:    SplitMetrics
    validation_sweep_rows: list[dict[str, float | int | str | None]]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    class_weight: str | dict[int, float] | None,
    max_iter: int,
    random_seed: int,
    solver: str,
) -> LogisticRegression:
    """Fit a logistic regression classifier.

    Inputs:
        x_train:      Training feature matrix, shape ``(n_train, n_features)``.
        y_train:      Binary training labels, shape ``(n_train,)``.
        c_value:      Inverse regularisation strength.  Smaller = stronger
                      regularisation.
        class_weight: ``None`` for uniform weighting; ``"balanced"`` to
                      up-weight the minority positive class proportionally to
                      the class imbalance.  "balanced" is strongly recommended
                      for this project because positives are rare.
        max_iter:     Maximum number of solver iterations.
        random_seed:  Random seed for the solver.
        solver:       Scikit-learn solver name.

    Outputs:
        Fitted ``LogisticRegression`` model.
    """
    model = LogisticRegression(
        C=c_value,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_seed,
        solver=solver,
    )
    model.fit(x_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Inference and evaluation
# ---------------------------------------------------------------------------

def predict_positive_probabilities(
    model: LogisticRegression,
    x: np.ndarray,
) -> np.ndarray:
    """Return the predicted P(positive) for each row in ``x``.

    Inputs:
        model: Fitted logistic regression model.
        x:     Feature matrix, shape ``(n_examples, n_features)``.

    Outputs:
        1-D float array of shape ``(n_examples,)``.
    """
    return model.predict_proba(x)[:, 1]


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> SplitMetrics:
    """Compute all metrics for thresholded predictions on a labelled split.

    Inputs:
        y_true:    Ground-truth binary labels.
        y_prob:    Predicted positive-class probabilities.
        threshold: Decision threshold; predictions >= threshold → positive.

    Outputs:
        Populated :class:`SplitMetrics` instance.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return SplitMetrics(
        threshold=float(threshold),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(fbeta_score(y_true, y_pred, beta=1.0, zero_division=0)),
        f2=float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        average_precision=float(average_precision_score(y_true, y_prob)),
        true_negative=int(tn),
        false_positive=int(fp),
        false_negative=int(fn),
        true_positive=int(tp),
        predicted_positive_count=int(y_pred.sum()),
        actual_positive_count=int(y_true.sum()),
    )


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float | int]]:
    """Evaluate metrics at every candidate threshold.

    Inputs:
        y_true:     Ground-truth labels for one split.
        y_prob:     Positive-class probabilities for the same split.
        thresholds: Candidate thresholds to evaluate.

    Outputs:
        List of metric dicts, one per threshold.
    """
    return [
        evaluate_predictions(y_true=y_true, y_prob=y_prob, threshold=t).to_dict()
        for t in thresholds
    ]


def select_threshold(
    sweep_rows: list[dict[str, float | int]],
) -> float:
    """Choose the probability threshold that maximises F2 score.

    F2 weights recall twice as heavily as precision, favouring models
    that miss fewer research mentions.  Ties are broken by higher recall,
    then by lower threshold.

    Inputs:
        sweep_rows: Output of :func:`sweep_thresholds`.

    Outputs:
        Selected threshold value.
    """
    best = max(
        sweep_rows,
        key=lambda r: (float(r["f2"]), float(r["recall"]), -float(r["threshold"])),
    )
    return float(best["threshold"])


# ---------------------------------------------------------------------------
# Full grid search
# ---------------------------------------------------------------------------

def select_best_logistic_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_values: list[float],
    class_weight_options: list[str | dict[int, float] | None],
    thresholds: list[float],
    max_iter: int,
    random_seed: int,
    solver: str,
) -> ModelSelectionResult:
    """Grid-search over C and class_weight, selecting the best model on validation.

    For each (C, class_weight) combination the function:
      1. Trains a logistic regression on the training set.
      2. Sweeps the candidate thresholds on the validation set.
      3. Selects the threshold that maximises F2 score.
      4. Records the resulting validation metrics.

    The winning configuration is the one whose validation metrics are best
    under the ordering: F2 > recall > precision > lower threshold.

    Inputs:
        x_train:              Training feature matrix.
        y_train:              Training labels.
        x_val:                Validation feature matrix.
        y_val:                Validation labels.
        c_values:             C candidates to sweep.
        class_weight_options: class_weight candidates (None and/or "balanced").
        thresholds:           Probability-threshold candidates.
        max_iter:             Maximum fit iterations per model.
        random_seed:          Seed for reproducibility.
        solver:               Logistic regression solver.

    Outputs:
        :class:`ModelSelectionResult` containing the winning model and all
        validation artifacts.

    Raises:
        RuntimeError: If no candidates were evaluated (empty sweep).
    """
    best: ModelSelectionResult | None = None

    for c_value in c_values:
        for class_weight in class_weight_options:

            # --- train ---
            model = train_logistic_regression(
                x_train=x_train,
                y_train=y_train,
                c_value=c_value,
                class_weight=class_weight,
                max_iter=max_iter,
                random_seed=random_seed,
                solver=solver,
            )

            # --- threshold sweep on validation ---
            y_val_prob    = predict_positive_probabilities(model, x_val)
            sweep_rows    = sweep_thresholds(y_true=y_val, y_prob=y_val_prob, thresholds=thresholds)
            best_threshold = select_threshold(sweep_rows=sweep_rows)
            val_metrics = evaluate_predictions(
                y_true=y_val,
                y_prob=y_val_prob,
                threshold=best_threshold,
            )

            # Annotate sweep rows with the hyper-params that produced them.
            annotated_rows = [
                {**row, "c_value": c_value, "class_weight": str(class_weight)}
                for row in sweep_rows
            ]

            candidate = ModelSelectionResult(
                model=model,
                best_c=c_value,
                best_class_weight=class_weight,
                best_threshold=best_threshold,
                validation_metrics=val_metrics,
                validation_sweep_rows=annotated_rows,
            )

            if best is None:
                best = candidate
                continue

            # Compare using the same F2-first ordering used for thresholds.
            def _rank(result):
                m = result.validation_metrics
                return (m.f2, m.recall, m.precision, -m.threshold)

            if _rank(candidate) > _rank(best): 
                best = candidate

    if best is None:
        raise RuntimeError(
            "Model selection evaluated zero candidates.  "
            "Check that c_values and class_weight_options are non-empty."
        )
    return best
