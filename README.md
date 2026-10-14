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

The controller only receives observations and public instrument calibration. Evaluator
records and hidden coefficients are saved separately. In-process interfaces prevent
accidental leakage; they are **not a security sandbox** against malicious Python code.
An isolated tool transport is required before third-party agent execution.
