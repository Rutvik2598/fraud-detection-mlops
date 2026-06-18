# fraud-detection-mlops — common tasks. `make help` lists targets.
# Python apps run from the local venv; Redpanda runs in docker-compose.

PY ?= python
TOPIC ?= transactions
PARTITIONS ?= 6
LIMIT ?= 50000
SPEED ?= 0

.PHONY: help install test lint format \
        train-baseline train-offline \
        redpanda-up redpanda-down redpanda-logs topic produce consume stream-demo

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
