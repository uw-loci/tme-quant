# Makefile for tme-quant
# Convenience commands for common tasks. The repository holds several
# independently built projects (pycurvelets-core and the backends);
# there is no project at the root, so targets loop over PROJECTS and use the
# shared .venv that bin/install.sh creates.

.PHONY: help setup test check clean wheel sdist dist

PROJECTS := pycurvelets-core pycurvelets-fdct pycurvelets-udct
PYTHON ?= $(CURDIR)/.venv/bin/python
DIST_DIR := dist

help:
	@echo "tme-quant Makefile Commands"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup        - Install core + FDCT and UDCT backends into .venv (uses bin/install.sh)"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run each project's test suite"
	@echo "  make check        - Run linting"
	@echo "  make clean        - Clean build artifacts"
	@echo ""
	@echo "Packaging (see README.md 'Wheel vs sdist'), output in $(DIST_DIR)/:"
	@echo "  make wheel        - Build a wheel per project"
	@echo "  make sdist        - Build a source dist per project"
	@echo "  make dist         - Both"
	@echo ""
	@echo "Projects: $(PROJECTS)"

setup:
	@echo "Running automated installation..."
	bash bin/install.sh

test:
	@set -e; for project in $(PROJECTS); do \
		echo "== pytest $$project"; \
		(cd $$project && $(PYTHON) -m pytest -v); \
	done

check:
	$(PYTHON) -m ruff check $(PROJECTS)

wheel:
	@set -e; for project in $(PROJECTS); do \
		$(PYTHON) -m build --wheel --outdir $(DIST_DIR) $$project; \
	done

sdist:
	@set -e; for project in $(PROJECTS); do \
		$(PYTHON) -m build --sdist --outdir $(DIST_DIR) $$project; \
	done

dist: sdist wheel

clean:
	rm -rf $(DIST_DIR) build/ */build/ */*.egg-info */src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache */.pytest_cache .coverage htmlcov
