"""End-to-end train/serve parity check through Feast (M3).

Builds the card-state snapshot, materializes it into the online store, then for a
sample of "live" transactions (the first one per card after the cutoff — the
realistic case: state is current, a new transaction arrives) compares:

  1. online velocity features (Feast get_online_features -> ODFV) vs the offline
     batch definition (add_velocity_features), and
  2. the full calibrated model score computed online vs offline.

Both must match exactly. Works against sqlite (no Docker) or Redis
(FEAST_ONLINE_STORE=redis). Exits non-zero on any mismatch.

Run:  python -m fraud_detection_mlops.serving.verify_parity --sample 300
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

import mlflow
import pandas as pd

from fraud_detection_mlops import config
from fraud_detection_mlops.data import load_training_data
from fraud_detection_mlops.features import VELOCITY_FEATURES, add_velocity_features
from fraud_detection_mlops.features.online import card_key
from fraud_detection_mlops.serving import store as store_mod

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("verify_parity")


def run(*, cutoff_fraction: float, sample: int, rebuild: bool) -> bool:
    if rebuild:
        store, cutoff = store_mod.setup_store(cutoff_fraction=cutoff_fraction)
    else:
        store = store_mod.get_store()
        df0 = load_training_data()
        cutoff = int(df0[config.TIME_COL].quantile(cutoff_fraction, interpolation="lower"))

    # Offline reference features for every transaction.
    df = load_training_data().sort_values([config.TIME_COL, config.ID_COL]).reset_index(drop=True)
    offline = add_velocity_features(df).set_index(config.ID_COL)

    # "Live" transactions: the first post-cutoff txn per card that has state.
    post = df[df[config.TIME_COL] > cutoff].copy()
    post["_ck"] = post.apply(card_key, axis=1)
    snap_cards = set(pd.read_parquet(config.CARD_STATE_PARQUET)["card_id"])
    post = post[post["_ck"].isin(snap_cards)]
    firsts = post.drop_duplicates("_ck", keep="first").head(sample)
    logger.info(
        "Checking %d live transactions (online store=%s)",
        len(firsts), store.config.online_store.type,
    )

    # Load the champion model for full score parity.
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(
        f"models:/{config.REGISTERED_MODEL_NAME}@{config.CHAMPION_ALIAS}"
    )
    cols = list(model.named_steps["preprocess"].feature_names_in_)

    feat_mismatch = 0
    score_mismatch = 0
    for _, r in firsts.iterrows():
        txn = r.drop(labels="_ck").to_dict()
        tid = txn[config.ID_COL]
        online = store_mod.online_velocity_features(store, txn)
        for f in VELOCITY_FEATURES:
            o, v = float(offline.loc[tid, f]), float(online[f])
            if not ((math.isnan(o) and math.isnan(v)) or abs(o - v) <= 1e-6 + 1e-6 * abs(o)):
                feat_mismatch += 1
                if feat_mismatch <= 10:
                    logger.error("FEATURE %s tid=%s online=%s offline=%s", f, tid, v, o)

        # Full score parity: online-assembled row vs offline-assembled row.
        online_row = {c: txn.get(c) for c in cols}
        online_row.update({f: online[f] for f in VELOCITY_FEATURES if f in online_row})
        online_df = pd.DataFrame([online_row], columns=cols)
        online_score = float(model.predict_proba(online_df)[0, 1])
        offline_score = float(model.predict_proba(offline.loc[[tid], cols])[0, 1])
        if abs(online_score - offline_score) > 1e-6:
            score_mismatch += 1
            if score_mismatch <= 10:
                logger.error(
                    "SCORE tid=%s online=%.8f offline=%.8f", tid, online_score, offline_score
                )

    n = len(firsts)
    if feat_mismatch or score_mismatch:
        logger.error(
            "FAILED: %d feature mismatches, %d score mismatches.", feat_mismatch, score_mismatch
        )
        return False
    logger.info(
        "PASS: %d live txns — all %d velocity features and all model scores match offline.",
        n, len(VELOCITY_FEATURES),
    )
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end Feast train/serve parity check.")
    p.add_argument("--cutoff-fraction", type=float, default=config.CARD_STATE_CUTOFF_FRACTION)
    p.add_argument("--sample", type=int, default=300)
    p.add_argument("--no-rebuild", action="store_true", help="Reuse an existing store.")
    args = p.parse_args()
    ok = run(cutoff_fraction=args.cutoff_fraction, sample=args.sample, rebuild=not args.no_rebuild)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
