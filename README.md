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

The controller only receives observations and public instrument calibration. Evaluator
records and hidden coefficients are saved separately. In-process interfaces prevent
accidental leakage; they are **not a security sandbox** against malicious Python code.
An isolated tool transport is required before third-party agent execution.

No source files from the existing private repository are copied into this prototype.
The standalone scalar propagator will serve as a small independent reference for a
future adapter to that repository. See the [release handoff](docs/release-handoff.md) for
the inspected reference releases and eventual packaging requirements.
