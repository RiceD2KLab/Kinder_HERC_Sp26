"""Model training, threshold selection, and evaluation utilities.

Workflow
--------
1. ``train_logistic_regression``   - fit a logistic regression model
2. ``train_xgboost``               - fit an XGBoost classifier
3. ``predict_positive_probabilities`` - get P(positive) for each example
4. ``sweep_thresholds``            - evaluate a grid of decision thresholds
5. ``select_threshold``            - pick the best threshold by F2 score
6. ``evaluate_predictions``        - compute all metrics at a chosen threshold
7. ``select_best_logistic_model``  - grid search over C and class_weight
8. ``select_best_xgboost_model``   - grid search over XGBoost hyperparameters
9. ``select_best_model``           - dispatcher: routes to LR or XGBoost

Threshold-selection philosophy
-------------------------------
The sponsor prefers *false positives over false negatives*: it is better to
flag extra chunks for review than to miss a genuine research mention.  The
pipeline selects the threshold that maximises the F2 score on the validation
set.  F2 weights recall twice as heavily as precision, favouring models that
miss fewer research mentions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

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

    Works for both logistic regression and XGBoost — the ``model`` field
    holds whatever fitted estimator was selected.  ``best_params`` always
    contains the full winning hyperparameter dict regardless of model type.
    ``best_c`` and ``best_class_weight`` are populated for logistic
    regression and None for XGBoost (kept for backward compatibility).

    Attributes:
        model:                 Fitted estimator (LR or XGBoost).
        best_c:                Winning C for logistic regression; None for XGBoost.
        best_class_weight:     Winning class_weight for LR; None for XGBoost.
        best_threshold:        Threshold selected on the validation set.
        validation_metrics:    Metrics of the winning model at best_threshold.
        validation_sweep_rows: All threshold-sweep rows (for the CSV artifact).
        best_params:           Full winning hyperparameter dict for any model type.
    """

    model:                 Any
    best_c:                float | None
    best_class_weight:     str | dict[int, float] | None
    best_threshold:        float
    validation_metrics:    SplitMetrics
    validation_sweep_rows: list[dict[str, float | int | str | None]]
    best_params:           dict = field(default_factory=dict)


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
# LASSO-based feature selection
# ---------------------------------------------------------------------------

def select_features_lasso(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    lasso_c_values: list[float],
    max_iter: int,
    random_seed: int,
    criterion: str = "bic",
) -> tuple[np.ndarray, float, int]:
    """Select features via LASSO, choosing the sparsity level by AIC or BIC
    evaluated on the validation set.

    For each candidate C value an L1-penalised logistic regression is fit on
    the training set.  The information criterion is then computed on the
    *validation* set so that feature selection uses held-out data:

        BIC = k · log(n_val) − 2 · LL_val
        AIC = 2k              − 2 · LL_val

    where k is the number of non-zero coefficients and LL_val is the
    log-likelihood of the fitted model evaluated on (x_val, y_val).
    Lower is better.  The C whose model achieves the lowest criterion score
    determines the final feature mask.

    Inputs:
        x_train:        Training feature matrix ``(n_train, n_features)``.
        y_train:        Binary training labels ``(n_train,)``.
        x_val:          Validation feature matrix ``(n_val, n_features)``.
        y_val:          Binary validation labels ``(n_val,)``.
        lasso_c_values: Candidate inverse-regularisation strengths.  Smaller
                        C → stronger L1 penalty → fewer selected features.
        max_iter:       Maximum solver iterations per fit.
        random_seed:    Reproducibility seed.
        criterion:      ``"bic"`` (default) or ``"aic"``.

    Outputs:
        Tuple of:
            feature_mask  — Boolean array of shape ``(n_features,)``.  True
                            for selected dimensions.
            best_lasso_c  — The C value whose model was chosen.
            n_selected    — Number of selected features (``mask.sum()``).

    Notes:
        * ``class_weight="balanced"`` is always used for the LASSO fit because
          the positive class is rare and an unweighted fit would drive most
          positive-class coefficients to zero.
        * If every C value zeroes out all features, the full feature set is
          returned as a fallback so the fold does not crash.
    """
    n_val = len(y_val)
    best_score: float = np.inf
    best_mask:  np.ndarray | None = None
    best_c:     float = lasso_c_values[0]

    for c in lasso_c_values:
        lasso = LogisticRegression(
            C=c,
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_seed,
        )
        lasso.fit(x_train, y_train)

        mask = np.abs(lasso.coef_[0]) > 0
        k = int(mask.sum())
        if k == 0:
            # All features zeroed out — skip this C; it's too aggressive.
            continue

        # Log-likelihood on the validation set.
        val_probs = np.clip(lasso.predict_proba(x_val)[:, 1], 1e-10, 1 - 1e-10)
        log_likelihood = float(
            np.sum(
                y_val * np.log(val_probs)
                + (1 - y_val) * np.log(1 - val_probs)
            )
        )

        if criterion == "bic":
            score = k * np.log(n_val) - 2.0 * log_likelihood
        else:  # aic
            score = 2.0 * k - 2.0 * log_likelihood

        if score < best_score:
            best_score = score
            best_mask  = mask.copy()
            best_c     = c

    if best_mask is None:
        # Fallback: every C was too aggressive.  Keep all features.
        best_mask = np.ones(x_train.shape[1], dtype=bool)

    return best_mask, best_c, int(best_mask.sum())


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
                best_params={"c": c_value, "class_weight": str(class_weight)},
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


# ---------------------------------------------------------------------------
# XGBoost training and grid search
# ---------------------------------------------------------------------------

def train_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    scale_pos_weight: float,
    random_seed: int,
    device: str = "cpu",
) -> Any:
    """Fit an XGBoost binary classifier.

    ``scale_pos_weight`` is set to n_negatives / n_positives by the caller,
    which compensates for the class imbalance in the same way that
    ``class_weight="balanced"`` does for logistic regression.

    Inputs:
        x_train:          Training feature matrix ``(n_train, n_features)``.
        y_train:          Binary training labels ``(n_train,)``.
        n_estimators:     Number of boosting rounds.
        max_depth:        Maximum tree depth.
        learning_rate:    Step size shrinkage (eta).
        scale_pos_weight: Weight ratio for positive class (n_neg / n_pos).
        random_seed:      Random seed for reproducibility.
        device:           ``"cpu"`` (default) or ``"cuda"`` for GPU training.

    Outputs:
        Fitted ``XGBClassifier`` instance.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError(
            "xgboost is not installed.  Run: pip install xgboost"
        )

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        eval_metric="logloss",
        verbosity=0,
        device=device,
    )
    model.fit(x_train, y_train)
    return model


def select_best_xgboost_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators_options: list[int],
    max_depth_options: list[int],
    learning_rate_options: list[float],
    thresholds: list[float],
    random_seed: int,
    device: str = "cpu",
) -> ModelSelectionResult:
    """Grid-search over XGBoost hyperparameters, selecting the best on validation.

    ``scale_pos_weight`` is computed automatically from the training labels
    as n_negatives / n_positives so class imbalance is always handled.

    The winning configuration is chosen by the same F2-first ordering used
    for logistic regression: F2 > recall > precision > lower threshold.

    Inputs:
        x_train:               Training feature matrix.
        y_train:               Training labels.
        x_val:                 Validation feature matrix.
        y_val:                 Validation labels.
        n_estimators_options:  Candidate numbers of boosting rounds.
        max_depth_options:     Candidate maximum tree depths.
        learning_rate_options: Candidate learning rates.
        thresholds:            Probability-threshold candidates.
        random_seed:           Seed for reproducibility.

    Outputs:
        :class:`ModelSelectionResult` with the winning XGBoost model.
    """
    # Compute class imbalance weight once from training labels.
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    best: ModelSelectionResult | None = None

    def _rank(result: ModelSelectionResult) -> tuple:
        m = result.validation_metrics
        return (m.f2, m.recall, m.precision, -m.threshold)

    for n_est in n_estimators_options:
        for max_d in max_depth_options:
            for lr in learning_rate_options:

                model = train_xgboost(
                    x_train=x_train,
                    y_train=y_train,
                    n_estimators=n_est,
                    max_depth=max_d,
                    learning_rate=lr,
                    scale_pos_weight=scale_pos_weight,
                    random_seed=random_seed,
                    device=device,
                )

                y_val_prob    = predict_positive_probabilities(model, x_val)
                sweep_rows    = sweep_thresholds(y_true=y_val, y_prob=y_val_prob, thresholds=thresholds)
                best_threshold = select_threshold(sweep_rows=sweep_rows)
                val_metrics   = evaluate_predictions(
                    y_true=y_val,
                    y_prob=y_val_prob,
                    threshold=best_threshold,
                )

                annotated_rows = [
                    {
                        **row,
                        "n_estimators": n_est,
                        "max_depth": max_d,
                        "learning_rate": lr,
                    }
                    for row in sweep_rows
                ]

                candidate = ModelSelectionResult(
                    model=model,
                    best_c=None,
                    best_class_weight=None,
                    best_threshold=best_threshold,
                    validation_metrics=val_metrics,
                    validation_sweep_rows=annotated_rows,
                    best_params={
                        "n_estimators":    n_est,
                        "max_depth":       max_d,
                        "learning_rate":   lr,
                        "scale_pos_weight": round(scale_pos_weight, 3),
                    },
                )

                if best is None or _rank(candidate) > _rank(best):
                    best = candidate

    if best is None:
        raise RuntimeError(
            "XGBoost model selection evaluated zero candidates.  "
            "Check that all option lists are non-empty."
        )
    return best


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def select_best_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: Any,
    thresholds: list[float],
    random_seed: int,
) -> ModelSelectionResult:
    """Route to the correct model selection function based on config.model_type.

    Inputs:
        x_train:    Training feature matrix.
        y_train:    Training labels.
        x_val:      Validation feature matrix.
        y_val:      Validation labels.
        config:     A ``ModelConfig`` instance (from config.py).
        thresholds: Probability-threshold candidates.
        random_seed: Seed for reproducibility.

    Outputs:
        :class:`ModelSelectionResult` from the winning model.

    Raises:
        ValueError: If ``config.model_type`` is not recognised.
    """
    if config.model_type == "logistic_regression":
        return select_best_logistic_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            c_values=config.c_values,
            class_weight_options=config.class_weight_options,
            thresholds=thresholds,
            max_iter=config.max_iter,
            random_seed=random_seed,
            solver=config.solver,
        )
    elif config.model_type == "xgboost":
        return select_best_xgboost_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            n_estimators_options=config.xgb_n_estimators_options,
            max_depth_options=config.xgb_max_depth_options,
            learning_rate_options=config.xgb_learning_rate_options,
            thresholds=thresholds,
            random_seed=random_seed,
            device=config.xgb_device,
        )
    else:
        raise ValueError(
            f"Unknown model_type '{config.model_type}'.  "
            "Choose 'logistic_regression' or 'xgboost'."
        )
