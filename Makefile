# Makefile for tme-quant
# Convenience commands for common tasks

.PHONY: help install install-dev test clean setup check

help:
	@echo "tme-quant Makefile Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup        - Automated installation (recommended; uses bin/install.sh)"
	@echo "  make install      - Install tme-quant package (requires active env)"
	@echo "  make install-dev  - Install with development dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run test suite"
	@echo "  make check        - Run linting"
	@echo "  make clean        - Clean build artifacts"
	@echo ""
	@echo "Docker (alternative; not validated):"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container"
	@echo "  make docker-stop     - Stop Docker container"
	@echo ""

setup:
	@echo "Running automated installation..."
	bash bin/install.sh

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest -v

check:
	pytest -v
	ruff check .

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

