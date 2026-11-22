.PHONY: figures report setup setup-local data eda-a evaluate predict monitor select tune tune-semi tune-tabpfn improve predict-self-training lint format test check

# Renku: run once per session (pip installs don't survive a shutdown).
# --no-deps so pip never replaces the conda packages from environment.yml.
setup:
	pip install --no-deps -e .
	pre-commit install

# Outside Renku (plain venv): pull runtime + dev deps from pyproject.toml
setup-local:
	pip install -e ".[dev]"
	pre-commit install

# Stage the challenge data into data/raw (add ARGS=--link to symlink instead of copy)
data:
	python -m brpmm.prepare $(ARGS)

# Process A analysis -> outputs/process_A/eda/REPORT.md (+ data/processed features)
eda-a:
	python scripts/eda_process_a.py $(ARGS)

# Every method on the shared CV protocol -> findings report
evaluate:
	python -m brpmm.evaluate --process A B

# LEADERBOARD FILE: final models on all calibration batches -> outputs/<time>_A-ensemble_B-pca_*.csv
# (never submits; submitting is brpmm.submission.submit)
predict:
	python -m brpmm.predict

# Accepted-only monitoring (the brief's second solution) -> outputs/<time>_monitor.csv/.json
monitor:
	python scripts/monitor.py

# Evidence for the submitted A model: final selection (~2 min), nested tuning, extra views
# and synthetic rejects (~30 min); TabPFN separately (pip install -e ".[tabpfn]", TABPFN_TOKEN)
select:
	python scripts/select_final_a.py
tune:
	python scripts/tune_process_a.py --part hpo
	python scripts/tune_process_a.py --part combine
	python scripts/tune_process_a.py --part clean
	python scripts/tune_process_a.py --part augment
tune-semi:
	python scripts/tune_process_a.py --part semi
	python scripts/tune_process_a.py --part semi_half
	python scripts/tune_process_a.py --part rules
	python scripts/tune_process_a.py --part rules_confirm
	python scripts/tune_process_a.py --part rules_half
tune-tabpfn:
	python scripts/tune_process_a.py --part tabpfn

# Feature engineering / time-series member inside the submitted ensemble (~10 min)
improve:
	python scripts/improve_process_a.py

# Leaderboard files with self-training on the unlabelled test batches (allowed by J&J)
figures:
	python scripts/figures.py
report:
	python scripts/batch_report.py
predict-self-training:
	python scripts/predict_self_training.py

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest

check: lint test
