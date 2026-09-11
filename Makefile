# Makefile for tme-quant
# Convenience commands for common tasks

.PHONY: help setup test check clean wheel sdist

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
	@echo "Packaging (see README.md 'Wheel vs sdist vs git-dev'):"
	@echo "  make wheel        - Build the runtime wheel (src/ only)"
	@echo "  make sdist        - Build the source dist (CI tests; no MATLAB dumps / viz PNGs)"
	@echo ""

setup:
	@echo "Running automated installation..."
	bash bin/install.sh

test:
	uv run pytest -v

check:
	uv run ruff check .

wheel:
	uv run python -m build --wheel

sdist:
	uv run python -m build --sdist

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
