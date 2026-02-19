# Makefile for tme-quant
# Convenience commands for common tasks

.PHONY: help setup test check clean

help:
	@echo "tme-quant Makefile Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup        - Automated installation with curvelops (uses bin/install.sh)"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run test suite"
	@echo "  make check        - Run linting"
	@echo "  make clean        - Clean build artifacts"
	@echo ""

setup:
	@echo "Running automated installation..."
	bash bin/install.sh

test:
	uv run pytest -v

check:
	uv run ruff check .

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
