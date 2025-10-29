# Makefile for tme-quant
# Convenience commands for common tasks

.PHONY: help install install-dev install-all test clean setup check

help:
	@echo "tme-quant Makefile Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup        - Automated setup (recommended for first-time users)"
	@echo "  make install      - Install tme-quant package"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make install-all  - Install with all optional dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run test suite"
	@echo "  make check        - Run linting and type checking"
	@echo "  make clean        - Clean build artifacts"
	@echo ""
	@echo "Docker (alternative method):"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container"
	@echo "  make docker-stop     - Stop Docker container"
	@echo ""

setup:
	@echo "Running automated setup..."
	bash bin/setup.sh

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

test:
	pytest -v

check:
	pytest -v --cov=curvealign_py --cov=ctfire_py
	ruff check .
	black --check .
	isort --check-only .

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov

docker-build:
	docker build -f docker/Dockerfile -t tme-quant:latest .

docker-run:
	docker-compose -f docker/docker-compose.yml up -d

docker-stop:
	docker-compose -f docker/docker-compose.yml down

