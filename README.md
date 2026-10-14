# ProbeBench — Noether research factory

This is the private `noetherlabs/probebench` repository for the new research programme.
It is separate from the existing `noether-ao` checkout and its ongoing work.
The website redesign belongs to Claude. No public release, model-API spending or cloud
compute is configured. No licence has been selected for publication.

The programme contains four studies sharing optical measurements, bounded actions,
evidence recording and independent evaluation:

1. **ProbeBench:** choose observations to resolve optical ambiguity and improve a star image.
2. **The Prediction Boundary:** finish the existing frozen controller study without changing
   its selection opportunities, reserved cohorts or scientific gates.
3. **When Recovery Hurts:** separate detection, intervention choice and transition effects.
4. **Which Twin Should We Trust?:** vary optical sampling while holding the plant and other
   physical factors fixed, then assess predictions of held-out interventions.

The first implementation is a **development-only** ProbeBench environment. It is not
a completed benchmark, final protocol, telescope result or agent evaluation. Its fixed
fixtures are for correctness and demonstration, not final held-out evidence.

See [PLAN.md](PLAN.md), [DECISIONS.md](DECISIONS.md), [STATUS.md](STATUS.md), and the
[architecture and test review](docs/engineering-review.md).

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

No source files from the existing private repository are copied into this prototype.
The standalone scalar propagator will serve as a small independent reference for a
future adapter to that repository. See the [release handoff](docs/release-handoff.md) for
the inspected reference releases and eventual packaging requirements.
