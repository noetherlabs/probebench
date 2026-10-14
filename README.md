# ProbeBench

The first implementation is a **development-only** ProbeBench environment. It is not
a completed benchmark, final protocol, telescope result or agent evaluation. Its fixed
fixtures are for correctness and demonstration, not final held-out evidence.

From a source checkout, using the pinned environment:

```sh
uv sync --locked --extra dev
uv run --locked python -m pytest -q
uv run --locked python -m probebench.run --output work/development-v1
uv run --locked python -m probebench.run --verify work/development-v1
```

The run directory is created exclusively; choose a new output name for a new run.
Every run includes a source snapshot, dependency lock, observations, evaluator records,
complete results, figure and generated development report. A missing manifest indicates
an interrupted run. Verification checks file integrity, not scientific generalization.

The original 16-episode demonstration was reproduced in a fresh local environment;
all 59 retained files matched exactly. See the [verification record](results/fresh-environment-reproduction-20261014.json).
To run its frozen source without writing interpreter caches into retained evidence:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=results/development-20261014/source/src \
  uv run --locked python -m probebench.run --output work/original-source-reproduction
uv run --locked python scripts/compare_reproduction.py \
  results/development-20261014 work/original-source-reproduction \
  --output work/reproduction-check.json
```

The [noise qualification](docs/noise-likelihood.md) compares calibrated-intensity least
squares with a Poisson–Gaussian likelihood on identical saved exposures. It has resumable
case records, complete optimizer-start diagnostics and a mathematical sum-truncation bound.
Its known development fixtures are separate from any future held-out benchmark.

```sh
uv run --locked python -m probebench.qualification \
  --config configs/noise_qualification_development.json --output work/noise-development
# Add --resume with the same source, configuration and runtime after interruption.
uv run --locked python -m probebench.qualification --verify work/noise-development
```

The controller only receives observations and public instrument calibration. Evaluator
records and hidden coefficients are saved separately. In-process interfaces prevent
accidental leakage; they are **not a security sandbox** against malicious Python code.
An isolated tool transport is required before third-party agent execution.
