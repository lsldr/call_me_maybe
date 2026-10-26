UV ?= uv
ARGS ?=
PYTEST_ARGS ?=
BENCH_ARGS ?=
DEVICE ?= cpu

.DEFAULT_GOAL := install
.PHONY: install run debug clean lint lint-strict test check

install:
	$(UV) sync --group dev

run:
	$(UV) run python -m src $(ARGS)

debug:
	$(UV) run python -m pdb -m src $(ARGS)

clean:
	@find . -type d \( -name .venv -o -name .git \) -prune -o \
		-type d \( -name __pycache__ -o -name .mypy_cache \
		-o -name .pytest_cache -o -name .ruff_cache \) \
		-prune -exec rm -rf {} +
	@rm -rf .uvsync .uvsync.lock

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

test:
	$(UV) run pytest $(PYTEST_ARGS)

check: lint-strict test
