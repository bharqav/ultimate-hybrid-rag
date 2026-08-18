.PHONY: install test lint format ci bench

install:
	pip install -r requirements.txt
	pip install ruff pytest mypy

test:
	pytest tests/

lint:
	ruff check .
	mypy .

format:
	ruff check --fix .
	ruff format .

ci: format lint test
