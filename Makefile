.PHONY: help install install-models test lint format demo toy clean

help:
	@echo "install         install the package (core deps only)"
	@echo "install-models  install the Qwen backbones as well (torch, transformers, ...)"
	@echo "test            run the test suite"
	@echo "lint            ruff + black --check"
	@echo "format          black"
	@echo "demo            run the toy walkthrough (no downloads, CPU only)"
	@echo "toy             run the full pipeline on the toy corpus via the CLI"
	@echo "clean           remove caches and build artefacts"

install:
	python -m pip install -e ".[dev]"

install-models:
	python -m pip install -e ".[dev,models]"

test:
	python -m pytest tests

lint:
	ruff check multiscore scripts tests examples
	black --check multiscore scripts tests examples

format:
	black multiscore scripts tests examples

demo:
	python examples/toy_demo.py

toy:
	python scripts/run_retrieval.py --config configs/toy.yaml

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
