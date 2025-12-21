.PHONY: setup install test lint format clean run-experiment app help

help:
	@echo "AutoMatLab Makefile Commands:"
	@echo "  make setup          - Install dependencies"
	@echo "  make install        - Install package in development mode"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linters (ruff)"
	@echo "  make format         - Format code (black)"
	@echo "  make run-experiment - Run experiment with default config"
	@echo "  make app            - Launch Streamlit dashboard"
	@echo "  make clean          - Clean generated files"

setup:
	pip install --upgrade pip
	pip install -e ".[dev]"

install:
	pip install -e .

test:
	pytest tests/ -v --cov=automatlabs --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/

format:
	black src/ tests/ scripts/ app/
	ruff check --fix src/ tests/ scripts/ app/

run-experiment:
	python -m automatlabs.run --config configs/default.yaml --method all

app:
	streamlit run app/streamlit_app.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pkl" -delete
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info


