"""Demonstrate the feedback loop end-to-end (M4).

Simulates the clock advancing through the timeline. At each step more delayed
labels mature, a challenger is retrained on the freshened data, and it is promoted
only if it beats the current champion on the fixed validation window. Then it
traces one specific late-arriving fraud label from "not usable yet" to "matured,
trained on, and part of a promoted model".

Run:  python -m fraud_detection_mlops.pipelines.demo_feedback
"""

from __future__ import annotations

import logging

import mlflow

from fraud_detection_mlops import config
from fraud_detection_mlops.pipelines import labels
from fraud_detection_mlops.pipelines.retrain_flow import retraining_flow

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("demo_feedback")
logger.setLevel(logging.INFO)


def _reset_feedback_model() -> None:
    """Delete the feedback registered model so the demo starts from a clean slate."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    try:
        client.delete_registered_model(config.FEEDBACK_MODEL_NAME)
        logger.info("Reset registered model %s", config.FEEDBACK_MODEL_NAME)
    except Exception:  # noqa: BLE001 — not present yet
        pass


def main() -> None:
    features = labels.load_features()  # builds the cache once if needed
    delay = config.LABEL_DELAY_SECONDS
    v_start = labels.val_start_dt(features)

    # Clocks: advance through the training region (before the validation window).
    q = features[config.TIME_COL].quantile
    clocks = [int(q(0.4)), int(q(0.6)), int(q(0.8))]

    _reset_feedback_model()

    print(f"\nLabel delay = {delay:,}s (~{delay // 86400}d). "
          f"Validation window starts at TransactionDT={v_start:,} (held out every round).\n")

    rows = []
    for i, clock in enumerate(clocks, 1):
        print(f"--- Round {i}: clock (now) = TransactionDT {clock:,} ---")
        s = retraining_flow(clock)  # flow reads the cached features internally
        rows.append((i, s))

    # Per-round table.
    print("\n=== Feedback loop: retraining rounds ===")
    header = (
        f"{'round':>5} {'clock':>12} {'matured':>9} {'n_train':>9} "
        f"{'challenger':>11} {'champion':>9} {'promoted':>9}"
    )
    print(header)
    print("-" * len(header))
    for i, s in rows:
        champ = f"{s['champion_pr_auc']:.4f}" if s["champion_pr_auc"] is not None else "   —"
        print(f"{i:>5} {s['clock']:>12,} {s['n_matured']:>9,} {s['n_train']:>9,} "
              f"{s['challenger_pr_auc']:>11.4f} {champ:>9} {'YES' if s['promoted'] else 'no':>9}")

    # Trace one late-arriving fraud label from "not yet usable" to "improves a model".
    store = labels.label_store(features)
    region = features[features[config.TIME_COL] < v_start]
    cand = region[region[config.TARGET] == 1].merge(
        store[[config.ID_COL, "label_available_dt"]], on=config.ID_COL
    )
    between = cand[
        (cand["label_available_dt"] > clocks[0]) & (cand["label_available_dt"] <= clocks[1])
    ]
    if not between.empty:
        t = between.iloc[len(between) // 2]
        print("\n=== Tracing one late label ===")
        print(f"TransactionID {int(t[config.ID_COL])}: fraud, occurred at TransactionDT "
              f"{int(t[config.TIME_COL]):,}.")
        print(f"  Its chargeback (label) only becomes usable at "
              f"{int(t['label_available_dt']):,} = occurred + {delay:,}s delay.")
        print(f"  Round 1 (clock {clocks[0]:,}): label NOT yet arrived "
              f"({int(t['label_available_dt']):,} > {clocks[0]:,}) -> excluded from training.")
        print(f"  Round 2 (clock {clocks[1]:,}): label has arrived "
              f"({int(t['label_available_dt']):,} <= {clocks[1]:,}) -> joined back, trained on.")
        r2 = rows[1][1]
        verdict = (
            f"improved PR-AUC {rows[0][1]['challenger_pr_auc']:.4f} -> "
            f"{r2['challenger_pr_auc']:.4f} and was PROMOTED to champion."
            if r2["promoted"] else "did not beat the champion this round."
        )
        print(f"  That round's challenger (with this and other freshly-matured labels) {verdict}")

    promotions = sum(1 for _, s in rows if s["promoted"])
    print(f"\nDone: {promotions}/{len(rows)} rounds promoted a better model from delayed labels.")


if __name__ == "__main__":
    main()
