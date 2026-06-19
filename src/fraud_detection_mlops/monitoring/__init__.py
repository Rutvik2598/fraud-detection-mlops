"""M5 monitoring: Evidently drift reports + Prometheus metrics.

The offline plane's watchdog: compare recent production traffic to the training
distribution, surface feature and prediction drift, expose serving + drift
metrics for Prometheus/Grafana, and trip the retraining flow when the world moves.
"""
