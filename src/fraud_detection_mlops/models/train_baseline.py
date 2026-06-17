"""M0 baseline training entrypoint.

Run:  python -m fraud_detection_mlops.models.train_baseline

Pipeline: load (LEFT JOIN identity) -> validate -> time-based split -> fit a
logistic-regression baseline on TRAIN ONLY -> score the later validation window
-> report PR-AUC / precision@k -> log everything to local MLflow.

Nothing here ever touches the test set (invariant 2). The pipeline is fit only
on the training split, so all imputation/scaling/encoding statistics come from
past data (invariant 1).
"""

from __future__ import annotations

import logging

import mlflow

from fraud_detection_mlops import config
from fraud_detection_mlops.data import load_training_data, time_based_split, validate_training_data
from fraud_detection_mlops.models import (
    build_baseline_pipeline,
    evaluate_scores,
    plot_pr_curve,
    select_feature_columns,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("train_baseline")


def main() -> dict[str, float]:
    # 1. Load + validate the labeled data.
    df = load_training_data()
    summary = validate_training_data(df)
    logger.info("Data summary: %s", summary)

    # 2. Time-based split (never random — invariant 2).
    train_df, val_df = time_based_split(df)
    del df  # free the full frame before fitting

    # 3. Feature typing + pipeline.
    numeric_features, categorical_features = select_feature_columns(train_df)
    pipeline = build_baseline_pipeline(numeric_features, categorical_features)

    feature_cols = numeric_features + categorical_features
    X_train, y_train = train_df[feature_cols], train_df[config.TARGET].to_numpy()
    X_val, y_val = val_df[feature_cols], val_df[config.TARGET].to_numpy()

    # 4. Fit on TRAIN ONLY, score the later validation window.
    logger.info("Fitting baseline on %d train rows...", len(X_train))
    pipeline.fit(X_train, y_train)
    val_scores = pipeline.predict_proba(X_val)[:, 1]

    # 5. Honest evaluation (PR-AUC headline, precision@k, no accuracy).
    metrics = evaluate_scores(y_val, val_scores)
    pr_curve_path = plot_pr_curve(y_val, val_scores, config.FIGURES_DIR / "baseline_pr_curve.png")

    # 6. Log to local MLflow (params, metrics, PR-curve artifact).
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(
            config.MLFLOW_EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACT_LOCATION
        )
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="m0_logreg_baseline"):
        mlflow.set_tags(
            {"milestone": "M0", "model_type": "logistic_regression", "stage": "baseline"}
        )
        mlflow.log_params(
            {
                "model": "LogisticRegression",
                "class_weight": "balanced",
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 2000,
                "seed": config.RANDOM_SEED,
                "split": "time_based",
                "train_fraction": config.TRAIN_FRACTION,
                "n_numeric_features": len(numeric_features),
                "n_categorical_features": len(categorical_features),
                "n_train": len(train_df),
                "n_val": len(val_df),
                "train_fraud_rate": round(float(y_train.mean()), 6),
                "val_fraud_rate": round(float(y_val.mean()), 6),
                "train_dt_max": int(train_df[config.TIME_COL].max()),
                "val_dt_min": int(val_df[config.TIME_COL].min()),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(pr_curve_path), artifact_path="figures")

    # 7. Console summary.
    logger.info("=== M0 BASELINE RESULTS (time-based validation) ===")
    logger.info("Validation fraud rate (base): %.4f", metrics["base_rate"])
    logger.info("PR-AUC (average precision):   %.4f  [HEADLINE]", metrics["pr_auc"])
    logger.info("Lift over base rate:          %.1fx", metrics["lift_over_base"])
    logger.info("ROC-AUC (reference only):     %.4f", metrics["roc_auc"])
    for k in (100, 500, 1000, 2000, 5000):
        logger.info(
            "precision@%-5d = %.3f   recall@%-5d = %.3f",
            k,
            metrics[f"precision_at_{k}"],
            k,
            metrics[f"recall_at_{k}"],
        )
    logger.info(
        "NOTE: %d val rows (%.1f%%) are saturated at p>=0.999 — the class-weighted "
        "LogReg is uncalibrated, so precision@k for k below that mass is tie-break "
        "noise, not skill. PR-AUC (curve-based) is unaffected. Calibration is M1.",
        int(metrics["n_scores_saturated"]),
        100 * metrics["frac_scores_saturated"],
    )
    return metrics


if __name__ == "__main__":
    main()
