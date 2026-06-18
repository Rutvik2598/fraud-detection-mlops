"""Gated registry promotion (M4).

A retrained challenger replaces the champion **only if it is actually better** on
the held-out validation window — PR-AUC, the same honest metric as M1 (invariant
3). The champion is re-scored on the *same* validation set as the challenger
(apples-to-apples), never compared on a stale logged number. If there is no
champion yet, the challenger becomes the first one.
"""

from __future__ import annotations

import logging

import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_detection_mlops import config

logger = logging.getLogger(__name__)


def pr_auc(pipeline, val_df: pd.DataFrame, model_cols: list[str]) -> float:
    """Validation PR-AUC for a fitted inference pipeline."""
    scores = pipeline.predict_proba(val_df[model_cols])[:, 1]
    return float(average_precision_score(val_df[config.TARGET].to_numpy(), scores))


def champion(registered_model_name: str):
    """Return the current champion model version, or None if unset."""
    client = mlflow.MlflowClient()
    try:
        return client.get_model_version_by_alias(registered_model_name, config.CHAMPION_ALIAS)
    except Exception:  # noqa: BLE001 — alias/model may not exist yet (first round)
        return None


def champion_pr_auc(
    registered_model_name: str, val_df: pd.DataFrame, model_cols: list[str]
) -> tuple[str | None, float | None]:
    """(version, PR-AUC) of the current champion on this validation set, or (None, None)."""
    mv = champion(registered_model_name)
    if mv is None:
        return None, None
    model = mlflow.sklearn.load_model(f"models:/{registered_model_name}@{config.CHAMPION_ALIAS}")
    return mv.version, pr_auc(model, val_df, model_cols)


def log_and_register(
    pipeline, *, params: dict, metrics: dict, val_df: pd.DataFrame, model_cols: list[str],
    registered_model_name: str, run_name: str,
) -> str:
    """Log the challenger to MLflow and register a new version; return its version."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.MLFLOW_FEEDBACK_EXPERIMENT) is None:
        mlflow.create_experiment(
            config.MLFLOW_FEEDBACK_EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACT_LOCATION
        )
    mlflow.set_experiment(config.MLFLOW_FEEDBACK_EXPERIMENT)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"milestone": "M4", "stage": "challenger"})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        example = val_df[model_cols].head(5)
        signature = mlflow.models.infer_signature(
            example, pipeline.predict_proba(example)[:, 1]
        )
        mlflow.sklearn.log_model(
            pipeline, name="model", signature=signature, input_example=example,
            registered_model_name=registered_model_name, serialization_format="cloudpickle",
        )
    client = mlflow.MlflowClient()
    version = max(
        client.search_model_versions(f"name='{registered_model_name}'"),
        key=lambda mv: int(mv.version),
    ).version
    return version


def should_promote(
    challenger_pr_auc: float, champion_pr_auc_value: float | None,
    *, margin: float = config.PROMOTION_MARGIN,
) -> bool:
    """Pure gate decision: promote a first champion, else require a real win."""
    if champion_pr_auc_value is None:
        return True
    return challenger_pr_auc > champion_pr_auc_value + margin


def gated_promote(
    registered_model_name: str, challenger_version: str,
    challenger_pr_auc: float, champion_pr_auc_value: float | None,
    *, margin: float = config.PROMOTION_MARGIN,
) -> bool:
    """Set the champion alias to the challenger iff it beats the incumbent.

    First champion (no incumbent) is promoted unconditionally; otherwise the
    challenger must beat the champion's validation PR-AUC by at least ``margin``.
    """
    client = mlflow.MlflowClient()
    promote = should_promote(challenger_pr_auc, champion_pr_auc_value, margin=margin)
    if champion_pr_auc_value is None:
        reason = "no incumbent — establishing first champion"
    else:
        reason = (
            f"challenger {challenger_pr_auc:.4f} "
            f"{'>' if promote else '<='} champion {champion_pr_auc_value:.4f} + margin {margin}"
        )
    if promote:
        client.set_registered_model_alias(
            registered_model_name, config.CHAMPION_ALIAS, challenger_version
        )
        logger.info("PROMOTED v%s to champion (%s)", challenger_version, reason)
    else:
        logger.info("REJECTED v%s (%s)", challenger_version, reason)
    return promote
