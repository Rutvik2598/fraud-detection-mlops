"""M1 offline model training entrypoint.

Run:  python -m fraud_detection_mlops.models.train_offline

Pipeline:
  load (LEFT JOIN identity)
    -> add point-in-time velocity features on the FULL timeline
    -> validate
    -> 3-way time split (train < calibration < validation)
    -> fit XGBoost on TRAIN ONLY, imbalance via scale_pos_weight
    -> isotonic-calibrate on the CALIBRATION slice
    -> select cost-minimizing threshold on CALIBRATION
    -> evaluate on VALIDATION (same window as M0) + apply the threshold
    -> log params/metrics/artifacts to MLflow and register the model.

Invariants enforced here: velocity features are point-in-time (invariant 1);
never touches the test set (invariant 2); PR-AUC headline, no accuracy
(invariant 3); calibrated probabilities + cost threshold (invariant 4); one
feature definition reused for train/serve via the serialized pipeline
(invariant 5); imbalance via scale_pos_weight, not resampling (invariant 6);
deterministic seeds (invariant 7).
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from xgboost import XGBClassifier

from fraud_detection_mlops import config
from fraud_detection_mlops.data import (
    load_training_data,
    time_based_split_three,
    validate_training_data,
)
from fraud_detection_mlops.features import add_velocity_features
from fraud_detection_mlops.models import (
    brier,
    evaluate_scores,
    operating_point,
    plot_calibration_curve,
    plot_cost_curve,
    plot_pr_curve,
    select_cost_threshold,
)
from fraud_detection_mlops.models.offline import XGB_PARAMS, fit_calibrated_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_offline")

# The M0 logistic-regression floor on the identical validation window. M1 must
# clearly beat this. (Reproduce with `python -m ...models.train_baseline`.)
M0_PR_AUC = 0.1839


def _feature_importance(preprocessor, booster: XGBClassifier, k: int = 20) -> pd.DataFrame:
    names = preprocessor.get_feature_names_out()
    gains = booster.feature_importances_
    df = pd.DataFrame({"feature": names, "gain_importance": gains})
    return df.sort_values("gain_importance", ascending=False).head(k).reset_index(drop=True)


def _plot_importance(importance: pd.DataFrame, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(importance["feature"][::-1], importance["gain_importance"][::-1])
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("Top features (M1 XGBoost)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> dict[str, float]:
    # 1. Load + engineer point-in-time features on the full timeline, then validate.
    df = load_training_data()
    df = add_velocity_features(df)
    summary = validate_training_data(df)
    logger.info("Data summary: %s", summary)

    # 2. Three-way time split. Validation == M0's window (comparable PR-AUC).
    train_df, calib_df, val_df = time_based_split_three(df)
    del df

    # 3-5. Fit preprocessing + XGBoost (scale_pos_weight) + isotonic calibration,
    #      all via the shared fitter so M1 and the M4 retraining flow are identical.
    fit = fit_calibrated_model(train_df, calib_df, seed=config.RANDOM_SEED)
    inference = fit["pipeline"]
    booster = fit["booster"]
    preprocess = fit["preprocess"]
    model_cols = fit["model_cols"]
    numeric, categorical = fit["numeric"], fit["categorical"]
    scale_pos_weight = fit["scale_pos_weight"]

    y_val = val_df[config.TARGET].to_numpy()
    amt_calib = calib_df[config.AMOUNT_COL].to_numpy()
    amt_val = val_df[config.AMOUNT_COL].to_numpy()

    # 6. Score validation (uncalibrated for the calibration comparison; calibrated
    #    for everything reported).
    val_scores_uncal = booster.predict_proba(preprocess.transform(val_df[model_cols]))[:, 1]
    val_scores = inference.predict_proba(val_df[model_cols])[:, 1]

    # 7. Honest evaluation on validation (PR-AUC headline; no accuracy).
    metrics = evaluate_scores(y_val, val_scores)
    metrics["brier_uncalibrated"] = brier(y_val, val_scores_uncal)
    metrics["brier_calibrated"] = brier(y_val, val_scores)
    metrics["m0_pr_auc"] = M0_PR_AUC
    metrics["pr_auc_lift_over_m0"] = metrics["pr_auc"] - M0_PR_AUC

    # 8. Cost-based threshold: select on CALIBRATION (calibrated scores), apply to VAL.
    y_calib = calib_df[config.TARGET].to_numpy()
    calib_scores = inference.predict_proba(calib_df[model_cols])[:, 1]
    thr = select_cost_threshold(y_calib, calib_scores, amt_calib)
    op = operating_point(y_val, val_scores, amt_val, thr["threshold"])
    metrics["chosen_threshold"] = thr["threshold"]
    metrics["val_cost_at_threshold"] = op["cost"]
    metrics["val_cost_block_none"] = float(amt_val[y_val == 1].sum())
    metrics["val_cost_block_all"] = config.COST_PER_FALSE_BLOCK * int((y_val == 0).sum())
    metrics["val_block_rate"] = op["block_rate"]
    metrics["val_precision_at_threshold"] = op["precision"]
    metrics["val_recall_at_threshold"] = op["recall"]
    metrics["val_fraud_dollars_recall"] = op["fraud_dollars_recall"]

    # 9. Artifacts.
    pr_path = plot_pr_curve(
        y_val, val_scores, config.FIGURES_DIR / "m1_pr_curve.png",
        title="M1 XGBoost precision-recall (time-based validation)",
    )
    cal_path = plot_calibration_curve(
        y_val, val_scores_uncal, val_scores, config.FIGURES_DIR / "m1_calibration.png"
    )
    cost_path = plot_cost_curve(
        y_val, val_scores, amt_val, thr["threshold"], config.FIGURES_DIR / "m1_cost_curve.png"
    )
    importance = _feature_importance(preprocess, booster)
    imp_csv = config.REPORTS_DIR / "m1_feature_importance.csv"
    imp_csv.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(imp_csv, index=False)
    imp_path = _plot_importance(importance, config.FIGURES_DIR / "m1_feature_importance.png")

    # 10. MLflow: params, metrics, artifacts, registered model + champion alias.
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.MLFLOW_OFFLINE_EXPERIMENT) is None:
        mlflow.create_experiment(
            config.MLFLOW_OFFLINE_EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACT_LOCATION
        )
    mlflow.set_experiment(config.MLFLOW_OFFLINE_EXPERIMENT)

    with mlflow.start_run(run_name="m1_xgboost_calibrated"):
        mlflow.set_tags(
            {
                "milestone": "M1",
                "model_type": "xgboost",
                "calibration": "isotonic",
                "stage": "offline",
            }
        )
        mlflow.log_params(
            {
                "model": "XGBClassifier",
                **{f"xgb_{k}": v for k, v in XGB_PARAMS.items()},
                "scale_pos_weight": round(scale_pos_weight, 4),
                "seed": config.RANDOM_SEED,
                "split": "time_based_3way",
                "model_train_fraction": config.MODEL_TRAIN_FRACTION,
                "train_fraction": config.TRAIN_FRACTION,
                "n_numeric_features": len(numeric),
                "n_categorical_features": len(categorical),
                "velocity_windows": list(config.VELOCITY_WINDOWS_SECONDS),
                "card_id_cols": list(config.CARD_ID_COLS),
                "cost_per_false_block": config.COST_PER_FALSE_BLOCK,
                "n_train": len(train_df),
                "n_calib": len(calib_df),
                "n_val": len(val_df),
            }
        )
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        for p in (pr_path, cal_path, cost_path, imp_path):
            mlflow.log_artifact(str(p), artifact_path="figures")
        mlflow.log_artifact(str(imp_csv), artifact_path="reports")

        signature = mlflow.models.infer_signature(val_df[model_cols].head(50), val_scores[:50])
        # cloudpickle: the pipeline wraps a custom transformer + XGBoost, which
        # MLflow's default skops serializer rejects as "untrusted types".
        mlflow.sklearn.log_model(
            inference,
            name="model",
            signature=signature,
            input_example=val_df[model_cols].head(5),
            registered_model_name=config.REGISTERED_MODEL_NAME,
            serialization_format="cloudpickle",
        )

    # Promote to champion alias (gated promotion logic lands in M4; for now the
    # freshly trained model becomes champion so downstream milestones can resolve it).
    client = mlflow.MlflowClient()
    latest = max(
        client.search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'"),
        key=lambda mv: int(mv.version),
    )
    client.set_registered_model_alias(
        config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS, latest.version
    )
    logger.info(
        "Registered %s v%s and set alias '%s'",
        config.REGISTERED_MODEL_NAME, latest.version, config.CHAMPION_ALIAS,
    )

    # 11. Console summary.
    logger.info("=== M1 OFFLINE RESULTS (time-based validation, same window as M0) ===")
    logger.info("PR-AUC (average precision):   %.4f  [HEADLINE]", metrics["pr_auc"])
    logger.info("  vs M0 baseline:             %.4f  (+%.4f, %.1f%% relative)",
                M0_PR_AUC, metrics["pr_auc_lift_over_m0"],
                100 * metrics["pr_auc_lift_over_m0"] / M0_PR_AUC)
    logger.info("ROC-AUC (reference only):     %.4f", metrics["roc_auc"])
    logger.info("Brier  uncalibrated -> calibrated: %.5f -> %.5f",
                metrics["brier_uncalibrated"], metrics["brier_calibrated"])
    logger.info("Saturated@p>=0.999: %d (%.2f%%)  [calibration removes the M0 score pile-up]",
                int(metrics["n_scores_saturated"]), 100 * metrics["frac_scores_saturated"])
    for k in (100, 500, 1000, 2000, 5000):
        logger.info("precision@%-5d = %.3f   recall@%-5d = %.3f",
                    k, metrics[f"precision_at_{k}"], k, metrics[f"recall_at_{k}"])
    logger.info("--- Cost-based decision (threshold chosen on calibration slice) ---")
    logger.info("Chosen block threshold:       %.4f", metrics["chosen_threshold"])
    logger.info("Val expected cost @ threshold: $%.0f", metrics["val_cost_at_threshold"])
    logger.info("  vs block-none: $%.0f | block-all: $%.0f",
                metrics["val_cost_block_none"], metrics["val_cost_block_all"])
    logger.info("Val block rate=%.3f  precision=%.3f  recall=%.3f  $-recall=%.3f",
                metrics["val_block_rate"], metrics["val_precision_at_threshold"],
                metrics["val_recall_at_threshold"], metrics["val_fraud_dollars_recall"])
    return metrics


if __name__ == "__main__":
    main()
