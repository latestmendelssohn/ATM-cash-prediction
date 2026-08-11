# ===========================================================================
# Developer convenience targets.  Run `make help` for the list.
# ===========================================================================
PYTHON ?= python
PKG := atmforecast

.PHONY: help venv install data pipeline forecast backtest test lint serve \
        docker-build docker-up docker-down clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## install the package + dependencies (editable)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -r requirements.txt

data:  ## generate the synthetic ATM dataset
	$(PYTHON) -m $(PKG).cli generate-data --out data/raw/atm_transactions.csv

pipeline:  ## run the full batch pipeline -> artifacts/
	$(PYTHON) scripts/run_pipeline.py --horizon 14 --service-level 0.95

forecast:  ## example: forecast ATM001 (override: make forecast ATM=ATM003)
	$(PYTHON) -m $(PKG).cli forecast --atm $(or $(ATM),ATM001) --model holt_winters

backtest:  ## example: rolling-origin leaderboard for ATM001
	$(PYTHON) -m $(PKG).cli backtest --atm $(or $(ATM),ATM001)

test:  ## run the pure-Python test suite
	$(PYTHON) -m pytest -q

lint:  ## static checks (best effort)
	-ruff check src tests
	-black --check src tests

serve:  ## run the FastAPI server locally
	uvicorn $(PKG).api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:  ## build the container image
	docker build -t atm-cash-forecasting:latest .

docker-up:  ## start the API via docker compose
	docker compose up --build

docker-down:  ## stop the compose stack
	docker compose down

clean:  ## remove caches and build artifacts
	rm -rf .pytest_cache **/__pycache__ *.egg-info build dist artifacts chroma_db
