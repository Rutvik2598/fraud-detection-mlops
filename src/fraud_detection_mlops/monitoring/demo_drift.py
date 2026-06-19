"""Demonstrate decay detection and recovery after retrain (M5), end-to-end.

  1. HEALTHY  — recent production looks like the training distribution: low drift,
     no alert; the champion scores well.
  2. DRIFT    — inject a covariate shift (the producer's --drift, applied to a
     window): feature + prediction drift fire, and the champion's PR-AUC decays.
  3. TRIGGER  — the drift-check flow detects it and fires the M4 retraining flow on
     the now-drifted data.
  4. RECOVERY — the retrained model scores the drifted population well again, and
     drift vs the updated reference falls back to healthy.

Runs in-process (no Docker). The live version streams drift via the producer and
shows the same on Grafana.

Run:  python -m fraud_detection_mlops.monitoring.demo_drift
"""

from __future__ import annotations

import logging

import mlflow
from sklearn.metrics import average_precision_score

from fraud_detection_mlops import config
from fraud_detection_mlops.monitoring import drift
from fraud_detection_mlops.pipelines import labels
from fraud_detection_mlops.pipelines.drift_flow import drift_check_flow

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("demo_drift")
logger.setLevel(logging.INFO)

N = config.DRIFT_WINDOW_SIZE


def _auc(model, df, cols) -> float:
    return float(average_precision_score(df[config.TARGET], model.predict_proba(df[cols])[:, 1]))


def main() -> None:
    cache = labels.load_features()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    champ = mlflow.sklearn.load_model(
        f"models:/{config.REGISTERED_MODEL_NAME}@{config.CHAMPION_ALIAS}"
    )
    cols = list(champ.named_steps["preprocess"].feature_names_in_)

    v_start = labels.val_start_dt(cache)
    train = cache[cache[config.TIME_COL] < v_start]
    val = cache[cache[config.TIME_COL] >= v_start]

    # Training distribution = the reference window.
    reference = drift.build_monitoring_frame(train.sample(N, random_state=1), champ, cols)
    report_dir = config.DRIFT_REPORT_DIR

    # 1. HEALTHY.
    healthy = drift.build_monitoring_frame(val.sample(N, random_state=2), champ, cols)
    s_healthy = drift.drift_report(reference, healthy, html_path=report_dir / "healthy.html")
    auc_healthy = _auc(champ, val, cols)

    # 2. DRIFT injected -> champion decays.
    drifted_val = drift.inject_drift_features(val)
    current = drift.build_monitoring_frame(drifted_val.sample(N, random_state=2), champ, cols)
    auc_decayed = _auc(champ, drifted_val, cols)

    # 3. TRIGGER: point the retrain at the now-drifted data and run the drift-check
    #    flow, which detects drift and fires the M4 retraining flow.
    drifted_cache = drift.inject_drift_features(cache)
    drifted_path = config.FEATURE_CACHE_PARQUET.with_name("feature_cache_drifted.parquet")
    drifted_cache.to_parquet(drifted_path, index=False)
    _reset_feedback_model()

    original_cache = config.FEATURE_CACHE_PARQUET
    config.FEATURE_CACHE_PARQUET = drifted_path  # retraining_flow reads this from disk
    try:
        result = drift_check_flow(
            reference, current, clock=int(v_start), html_path=report_dir / "drifted.html"
        )
    finally:
        config.FEATURE_CACHE_PARQUET = original_cache

    # 4. RECOVERY: the retrained champion on the drifted population.
    new_model = mlflow.sklearn.load_model(
        f"models:/{config.FEEDBACK_MODEL_NAME}@{config.CHAMPION_ALIAS}"
    )
    new_ref = drift.build_monitoring_frame(
        drift.inject_drift_features(train).sample(N, random_state=1), new_model, cols
    )
    new_cur = drift.build_monitoring_frame(drifted_val.sample(N, random_state=2), new_model, cols)
    s_recovered = drift.drift_report(new_ref, new_cur, html_path=report_dir / "recovered.html")
    auc_recovered = _auc(new_model, drifted_val, cols)

    # Report.
    print("\n=== Drift monitoring: decay detection + recovery ===")
    hdr = (
        f"{'state':<26}{'feat_drift_share':>17}{'dataset_drift':>15}"
        f"{'pred_drift':>12}{'PR-AUC':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    print(f"{'1. healthy':<26}{s_healthy['feature_drift_share']:>17}"
          f"{str(s_healthy['dataset_drift']):>15}{str(s_healthy['prediction_drift']):>12}{auc_healthy:>9.4f}")
    print(f"{'2. drift injected (champ)':<26}{result['feature_drift_share']:>17}"
          f"{str(result['dataset_drift']):>15}{str(result['prediction_drift']):>12}{auc_decayed:>9.4f}")
    print(f"{'4. after retrain (new)':<26}{s_recovered['feature_drift_share']:>17}"
          f"{str(s_recovered['dataset_drift']):>15}{str(s_recovered['prediction_drift']):>12}{auc_recovered:>9.4f}")
    print(f"\nStep 3: drift-check flow detected drift and retrained = {result['retrained']}.")
    print(f"Decay: PR-AUC {auc_healthy:.4f} -> {auc_decayed:.4f} under drift; "
          f"recovered to {auc_recovered:.4f} after retrain.")
    print(f"Evidently HTML reports in {report_dir}/")


def _reset_feedback_model() -> None:
    try:
        mlflow.MlflowClient().delete_registered_model(config.FEEDBACK_MODEL_NAME)
    except Exception:  # noqa: BLE001 — not present yet
        pass


if __name__ == "__main__":
    main()
