"""Prefect retraining flow (M4).

One retrain at a simulated clock ``available_until_dt``:
  build matured-label training data -> fit a calibrated challenger ->
  score it and the current champion on the held-out validation window ->
  log + register the challenger -> promote it to champion only if it wins.

Run a single round:
  python -m fraud_detection_mlops.pipelines.retrain_flow --clock 8000000
"""

from __future__ import annotations

import logging

import pandas as pd
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from fraud_detection_mlops import config
from fraud_detection_mlops.models.offline import fit_calibrated_model
from fraud_detection_mlops.pipelines import labels, promote

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("retrain_flow")


@task(cache_policy=NO_CACHE)
def build_training_data(features: pd.DataFrame, clock: int) -> dict:
    return labels.build_training_data(features, clock)


@task(cache_policy=NO_CACHE)
def train_challenger(train_df: pd.DataFrame, calib_df: pd.DataFrame) -> dict:
    fit = fit_calibrated_model(train_df, calib_df)
    return {
        "pipeline": fit["pipeline"],
        "model_cols": fit["model_cols"],
        "scale_pos_weight": fit["scale_pos_weight"],
    }


@task(cache_policy=NO_CACHE)
def evaluate_models(
    pipeline, val_df: pd.DataFrame, model_cols: list[str], registered_model_name: str
) -> dict:
    challenger = promote.pr_auc(pipeline, val_df, model_cols)
    champ_version, champ_auc = promote.champion_pr_auc(registered_model_name, val_df, model_cols)
    return {
        "challenger_pr_auc": challenger,
        "champion_version": champ_version,
        "champion_pr_auc": champ_auc,
    }


@task(cache_policy=NO_CACHE)
def register_and_gate(
    pipeline, val_df: pd.DataFrame, model_cols: list[str], registered_model_name: str,
    params: dict, metrics: dict, run_name: str, challenger_auc: float, champ_auc: float | None,
) -> dict:
    version = promote.log_and_register(
        pipeline, params=params, metrics=metrics, val_df=val_df, model_cols=model_cols,
        registered_model_name=registered_model_name, run_name=run_name,
    )
    promoted = promote.gated_promote(registered_model_name, version, challenger_auc, champ_auc)
    return {"version": version, "promoted": promoted}


@flow(name="fraud-retraining")
def retraining_flow(
    available_until_dt: int,
    *,
    features: pd.DataFrame | None = None,
    registered_model_name: str = config.FEEDBACK_MODEL_NAME,
) -> dict:
    """Retrain + gated-promote for one simulated clock; return a round summary."""
    if features is None:
        features = labels.load_features()

    data = build_training_data(features, available_until_dt)
    fit = train_challenger(data["train_df"], data["calib_df"])
    evald = evaluate_models(
        fit["pipeline"], data["val_df"], fit["model_cols"], registered_model_name
    )

    params = {
        "clock": available_until_dt,
        "label_delay_seconds": config.LABEL_DELAY_SECONDS,
        "n_matured": data["n_matured"],
        "n_train": len(data["train_df"]),
        "n_calib": len(data["calib_df"]),
        "n_val": len(data["val_df"]),
        "scale_pos_weight": round(fit["scale_pos_weight"], 4),
        "val_fraction": config.VAL_FRACTION,
        "registered_model_name": registered_model_name,
    }
    champ_auc = evald["champion_pr_auc"]
    metrics = {
        "pr_auc": evald["challenger_pr_auc"],
        "champion_pr_auc": champ_auc if champ_auc is not None else -1.0,
    }
    gate = register_and_gate(
        fit["pipeline"], data["val_df"], fit["model_cols"], registered_model_name,
        params, metrics, f"retrain_clock_{available_until_dt}",
        evald["challenger_pr_auc"], evald["champion_pr_auc"],
    )

    summary = {
        "clock": available_until_dt,
        "n_matured": data["n_matured"],
        "n_train": len(data["train_df"]),
        "challenger_version": gate["version"],
        "challenger_pr_auc": round(evald["challenger_pr_auc"], 4),
        "champion_version": evald["champion_version"],
        "champion_pr_auc": round(evald["champion_pr_auc"], 4) if evald["champion_pr_auc"] else None,
        "promoted": gate["promoted"],
    }
    logger.info("Round summary: %s", summary)
    return summary


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run one retraining round at a simulated clock.")
    p.add_argument("--clock", type=int, required=True, help="available_until_dt (TransactionDT).")
    p.add_argument("--model-name", default=config.FEEDBACK_MODEL_NAME)
    args = p.parse_args()
    retraining_flow(args.clock, registered_model_name=args.model_name)


if __name__ == "__main__":
    main()
