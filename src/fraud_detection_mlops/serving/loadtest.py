"""Latency load test for the scoring service (M3).

Samples real post-cutoff transactions, warms the service, then fires a fixed
number of requests at a target concurrency and reports p50/p99 latency — both
server-side processing time (the <50ms budget) and client round-trip — plus
throughput. Warmup matters: the first request pays one-off costs (Feast registry
load, model JIT) that don't reflect steady state.

Run (service must be up):  python -m fraud_detection_mlops.serving.loadtest -n 2000 -c 16
"""

from __future__ import annotations

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from fraud_detection_mlops import config
from fraud_detection_mlops.data import load_training_data
from fraud_detection_mlops.features.online import card_key


def _payloads(cutoff_fraction: float, n: int) -> list[dict]:
    """Sample post-cutoff transactions whose card has materialized state."""
    df = load_training_data().sort_values([config.TIME_COL, config.ID_COL]).reset_index(drop=True)
    cutoff = int(df[config.TIME_COL].quantile(cutoff_fraction, interpolation="lower"))
    post = df[df[config.TIME_COL] > cutoff]
    snap = set(pd.read_parquet(config.CARD_STATE_PARQUET)["card_id"])
    post = post[post.apply(lambda r: card_key(r) in snap, axis=1)].head(n)
    def _clean(v):
        if isinstance(v, float) and math.isnan(v):
            return None  # NaN -> null for valid JSON
        return v.item() if hasattr(v, "item") else v

    return [{k: _clean(v) for k, v in row.to_dict().items()} for _, row in post.iterrows()]


def _percentiles(values: list[float]) -> dict[str, float]:
    a = np.array(values)
    return {
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def run(url: str, n_requests: int, concurrency: int, cutoff_fraction: float, warmup: int) -> None:
    payloads = _payloads(cutoff_fraction, max(n_requests, warmup))
    if not payloads:
        raise SystemExit("No payloads sampled — is the store built?")
    print(f"Sampled {len(payloads)} payloads. Warming up ({warmup})...")
    for i in range(warmup):
        requests.post(url, json=payloads[i % len(payloads)], timeout=30)

    server_ms: list[float] = []
    client_ms: list[float] = []
    errors = 0

    def _one(i: int):
        payload = payloads[i % len(payloads)]
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=payload, timeout=30)
            client = (time.perf_counter() - t0) * 1000
            j = r.json()
            return j.get("latency_ms"), client, r.status_code == 200
        except Exception:
            return None, (time.perf_counter() - t0) * 1000, False

    print(f"Firing {n_requests} requests at concurrency {concurrency}...")
    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for srv, cli, ok in pool.map(_one, range(n_requests)):
            if ok and srv is not None:
                server_ms.append(srv)
                client_ms.append(cli)
            else:
                errors += 1
    wall = time.perf_counter() - wall0

    sp, cp = _percentiles(server_ms), _percentiles(client_ms)
    budget = config.LATENCY_BUDGET_MS
    print("\n=== Latency under load ===")
    print(f"requests={len(server_ms)}  errors={errors}  concurrency={concurrency}")
    print(f"throughput = {len(server_ms) / wall:.0f} req/s over {wall:.1f}s")
    print(f"server-side processing ms:  p50={sp['p50']:.2f}  p99={sp['p99']:.2f}  "
          f"max={sp['max']:.2f}  mean={sp['mean']:.2f}")
    print(f"client round-trip ms:       p50={cp['p50']:.2f}  p99={cp['p99']:.2f}  "
          f"max={cp['max']:.2f}  mean={cp['mean']:.2f}")
    verdict = "PASS" if sp["p99"] < budget else "OVER BUDGET"
    print(f"budget = {budget:.0f}ms (server-side)  ->  p99 {sp['p99']:.2f}ms  [{verdict}]")


def main() -> None:
    p = argparse.ArgumentParser(description="Scoring service latency load test.")
    p.add_argument("--url", default=f"http://127.0.0.1:{config.SERVING_PORT}/score")
    p.add_argument("-n", "--n-requests", type=int, default=2000)
    p.add_argument("-c", "--concurrency", type=int, default=16)
    p.add_argument("--cutoff-fraction", type=float, default=config.CARD_STATE_CUTOFF_FRACTION)
    p.add_argument("--warmup", type=int, default=50)
    args = p.parse_args()
    run(args.url, args.n_requests, args.concurrency, args.cutoff_fraction, args.warmup)


if __name__ == "__main__":
    main()
