UV ?= uv
ARGS ?=
PYTEST_ARGS ?=
BENCH_ARGS ?=
DEVICE ?= cpu
TRACE ?=

.DEFAULT_GOAL := install
.PHONY: install run debug clean lint lint-strict test check bench demo visualize

install:
	$(UV) sync --group dev

run:
	$(UV) run $(if $(findstring --visual,$(ARGS)),--extra visualization,) python -m src $(ARGS)

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
	$(UV) run $(if $(findstring --visual,$(ARGS) $(PYTEST_ARGS)),--extra visualization,) pytest $(if $(findstring --visual,$(ARGS) $(PYTEST_ARGS)),-s,) $(ARGS) $(PYTEST_ARGS)

check: lint-strict test

bench:
	$(UV) run $(if $(findstring --visual,$(BENCH_ARGS)),--extra visualization,) python -m src.evaluation.benchmark --device $(DEVICE) \
		--output benchmarks/bonus/benchmark $(BENCH_ARGS)

demo:
	$(UV) run python -m src.evaluation.demo --device $(DEVICE) \
		--output benchmarks/bonus/demonstration $(ARGS)

visualize:
	$(UV) run --extra visualization python -m src.visualization $(TRACE) $(ARGS)
