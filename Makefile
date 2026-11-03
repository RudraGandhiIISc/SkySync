# Ask IISc gate. Run targets inside Slurm, never on the login node:
#   sbatch --wait slurm/gate.sh <target>
# slurm/lib.sh sets UV_PROJECT_ENVIRONMENT, so `uv run` uses the scratch venv.

UV ?= uv
RUN := $(UV) run
SRC := src tests eval tools deploy
COV_100 := src/askiisc/access src/askiisc/gateway/limits.py \
	src/askiisc/gateway/drr.py src/askiisc/autoscaler/policy.py

.PHONY: fmt fmt-check lint type test test-int test-e2e test-perf test-sec \
	test-regress nightly audit hooks all

fmt:
	$(RUN) ruff format $(SRC)
	$(RUN) ruff check --fix $(SRC)

fmt-check:
	$(RUN) ruff format --check $(SRC)

lint:
	$(RUN) ruff check $(SRC)

type:
	$(RUN) mypy

test:
	$(RUN) pytest -q $(PYTEST_ARGS) tests/unit tests/conformance \
		--cov=askiisc --cov-report=term --cov-report=json:build/coverage.json \
		--cov-fail-under=90
	$(RUN) python tools/check_coverage.py build/coverage.json $(COV_100)

test-int:
	$(RUN) pytest -q -m integration tests/integration

test-e2e:
	$(RUN) pytest -q -m e2e tests/e2e

test-perf:
	$(RUN) pytest -q -m perf tests/perf

test-sec:
	$(RUN) bandit -q -r src -c pyproject.toml
	$(RUN) pytest -q -m security tests/security
	@if command -v gitleaks >/dev/null; then \
		gitleaks detect --source . --no-banner --redact; \
	else echo "gitleaks not installed: skipped"; fi

test-regress:
	$(RUN) pytest -q -m regress tests/regression

# Before merging a phase (no cron here): sbatch slurm/gate.sh nightly
nightly: test-perf test-sec test-regress

audit:
	@mkdir -p build
	$(UV) export --frozen --no-emit-project --format requirements-txt \
		> build/requirements.txt
	$(RUN) pip-audit --disable-pip --no-deps --require-hashes \
		-r build/requirements.txt

hooks:
	$(RUN) pre-commit run --all-files

all: fmt-check lint type test audit
