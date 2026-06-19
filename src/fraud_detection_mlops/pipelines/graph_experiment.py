"""Do fraud-ring graph features improve detection? An honest before/after.

Trains two models on the identical time-based split -- one with the existing
features, one with the graph features added -- and compares PR-AUC and
precision@k on the same held-out validation window. Logs both to MLflow and
registers the graph model as a challenger, gate-promoted via the feedback-loop
machinery.

The graph model is registered under its own name, not the serving champion:
serving these features would mean wiring them into the online aggregator and
feature store first, the same path the velocity features took from batch to
online. This proves the detection lift; productionizing the serving is the
follow-on.

Run:  python -m fraud_detection_mlops.pipelines.graph_experiment
"""

from __future__ import annotations

import logging

import mlflow

from fraud_detection_mlops import config
from fraud_detection_mlops.data import time_based_split_three
from fraud_detection_mlops.features import GRAPH_FEATURES, add_graph_features
from fraud_detection_mlops.models.evaluate import evaluate_scores
from fraud_detection_mlops.models.offline import fit_calibrated_model, score_frame
from fraud_detection_mlops.pipelines import labels, promote

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("graph_experiment")

GRAPH_MODEL_NAME = "fraud-detection-graph"


def _fit_eval(train_df, calib_df, val_df) -> tuple[float, dict, object, list[str]]:
    fit = fit_calibrated_model(train_df, calib_df)
    cols = fit["model_cols"]
    scores = score_frame(fit["pipeline"], val_df, cols)
    metrics = evaluate_scores(val_df[config.TARGET].to_numpy(), scores)
    return metrics["pr_auc"], metrics, fit, cols


def main() -> None:
    # Reuse the cached velocity features; add the graph features on top.
    features = add_graph_features(labels.load_features())
    train_df, calib_df, val_df = time_based_split_three(features)

    # Baseline = the identical pipeline without the graph features.
    base_train = train_df.drop(columns=list(GRAPH_FEATURES))
    base_calib = calib_df.drop(columns=list(GRAPH_FEATURES))
    base_val = val_df.drop(columns=list(GRAPH_FEATURES))

    logger.info("Training baseline (no graph features)...")
    base_auc, base_m, _, _ = _fit_eval(base_train, base_calib, base_val)
    logger.info("Training graph model (+ %d ring features)...", len(GRAPH_FEATURES))
    graph_auc, graph_m, graph_fit, graph_cols = _fit_eval(train_df, calib_df, val_df)

    # Did the model actually use the graph features?
    names = graph_fit["preprocess"].get_feature_names_out()
    gains = dict(zip(names, graph_fit["booster"].feature_importances_, strict=False))
    graph_importance = {f: round(float(gains.get(f, 0.0)), 5) for f in GRAPH_FEATURES}
    ranked = sorted(gains, key=gains.get, reverse=True)
    in_top = [f for f in ranked[:40] if f in GRAPH_FEATURES]

    # Register the graph model and gate-promote within its own registry name.
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    _, champ_auc = promote.champion_pr_auc(GRAPH_MODEL_NAME, val_df, graph_cols)
    version = promote.log_and_register(
        graph_fit["pipeline"],
        params={"model": "xgboost+graph", "n_features": len(graph_cols),
                "graph_features": list(GRAPH_FEATURES)},
        metrics={"pr_auc": graph_auc, "baseline_pr_auc": base_auc},
        val_df=val_df, model_cols=graph_cols,
        registered_model_name=GRAPH_MODEL_NAME, run_name="graph_model",
    )
    promoted = promote.gated_promote(GRAPH_MODEL_NAME, version, graph_auc, champ_auc)

    lift = graph_auc - base_auc
    print("\n=== Fraud-ring graph features: before / after ===")
    print(f"{'metric':<22}{'baseline':>12}{'+ graph':>12}{'delta':>12}")
    print("-" * 58)
    print(f"{'PR-AUC':<22}{base_auc:>12.4f}{graph_auc:>12.4f}{lift:>+12.4f}")
    print(f"{'ROC-AUC':<22}{base_m['roc_auc']:>12.4f}{graph_m['roc_auc']:>12.4f}"
          f"{graph_m['roc_auc'] - base_m['roc_auc']:>+12.4f}")
    for k in (500, 1000, 2000, 5000):
        b, g = base_m[f"precision_at_{k}"], graph_m[f"precision_at_{k}"]
        print(f"{'precision@' + str(k):<22}{b:>12.3f}{g:>12.3f}{g - b:>+12.3f}")
    print(f"\nPR-AUC lift: {lift:+.4f} ({100 * lift / base_auc:+.1f}% relative).")
    print(f"Graph features in top-40 importances: {in_top or 'none'}")
    print(f"Per-feature gain importance: {graph_importance}")
    print(f"Registered {GRAPH_MODEL_NAME} v{version} (promoted={promoted}).")


if __name__ == "__main__":
    main()
