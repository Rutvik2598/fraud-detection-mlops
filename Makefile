# fraud-detection-mlops — common tasks. `make help` lists targets.
# Python apps run from the local venv; Redpanda runs in docker-compose.

PY ?= python
TOPIC ?= transactions
PARTITIONS ?= 6
LIMIT ?= 50000
SPEED ?= 0
WORKERS ?= 4
PORT ?= 8001

.PHONY: help install test lint format \
        train-baseline train-offline \
        redpanda-up redpanda-down redpanda-logs topic produce consume stream-demo \
        store-up feast-build serve score-verify loadtest

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create venv deps (editable install with dev extras) via uv
	uv pip install -e ".[dev]"

test: ## Run the test suite
	$(PY) -m pytest -q

lint: ## Lint with ruff
	ruff check src/ tests/

format: ## Auto-format with ruff
	ruff format src/ tests/ && ruff check --fix src/ tests/

train-baseline: ## M0: train + log the logistic baseline
	$(PY) -m fraud_detection_mlops.models.train_baseline

train-offline: ## M1: train XGBoost + calibrate + register
	$(PY) -m fraud_detection_mlops.models.train_offline

redpanda-up: ## M2: start Redpanda + Console
	docker compose up -d
	@echo "Console: http://localhost:8080"

redpanda-down: ## M2: stop Redpanda + Console (keeps data volume)
	docker compose down

redpanda-logs: ## M2: tail Redpanda logs
	docker compose logs -f redpanda

topic: ## M2: create the transactions topic with PARTITIONS partitions
	docker compose exec -T redpanda rpk topic create $(TOPIC) -p $(PARTITIONS) || true
	docker compose exec -T redpanda rpk topic describe $(TOPIC)

produce: ## M2: replay LIMIT txns at SPEED (0=flood) onto the topic
	$(PY) -m fraud_detection_mlops.streaming.producer --limit $(LIMIT) --speed $(SPEED) --topic $(TOPIC)

consume: ## M2: consume + update rolling features (writes reports/stream_features.jsonl)
	$(PY) -m fraud_detection_mlops.streaming.consumer --topic $(TOPIC) --output reports/stream_features.jsonl

stream-demo: ## M2: end-to-end check — produce a slice, consume it, verify parity vs offline
	$(PY) -m fraud_detection_mlops.streaming.verify --limit $(LIMIT) --topic $(TOPIC)

store-up: ## M3: start Redis (the Feast online store)
	docker compose up -d redis
	@echo "Redis up on localhost:6379"

feast-build: ## M3: build card-state snapshot, apply + materialize into the online store
	$(PY) -m fraud_detection_mlops.serving.store

serve: ## M3: run the FastAPI scoring service ($(WORKERS) workers on port $(PORT))
	$(PY) -m uvicorn fraud_detection_mlops.serving.app:app --host 0.0.0.0 --port $(PORT) --workers $(WORKERS)

score-verify: ## M3: prove train/serve parity (online features + scores == offline)
	$(PY) -m fraud_detection_mlops.serving.verify_parity --sample 300

loadtest: ## M3: measure p50/p99 scoring latency under load
	$(PY) -m fraud_detection_mlops.serving.loadtest -n 4000 -c $(WORKERS)
