.PHONY: setup setup-local lint format test check

# Renku: run once per session (pip installs don't survive a shutdown).
# --no-deps so pip never replaces the conda packages from environment.yml.
setup:
	pip install --no-deps -e .
	pre-commit install

# Outside Renku (plain venv): pull runtime + dev deps from pyproject.toml
setup-local:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest

check: lint test
