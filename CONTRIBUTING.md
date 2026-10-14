# Contributing

ProbeBench changes should preserve the optical model, information boundary and experiment
accounting documented in [the model specification](docs/optical-model.md).

Before submitting a change:

```sh
uv sync --locked --extra dev
uv run --locked python -m pytest -q
uv run --locked python -m probebench.run --verify results/development-20261014
uv run --locked python -m probebench.qualification --verify results/noise-qualification-20261014
uv run --locked python -m probebench.sequential --verify results/sequential-calibration-20261014
uv run --locked python -m probebench.sequential --verify results/measurement-schedule-20261014
python3 scripts/check_repository.py
gitleaks git . --redact --no-banner
```

Use focused technical commits describing the changed behavior. Changes to propagation,
camera statistics, likelihoods or budget semantics require independent numerical or
behavioral checks. Report limitations and retain failed or non-winning outcomes.

Enable the repository's local push checks after installing Gitleaks:

```sh
git config --local core.hooksPath .githooks
```

The hook checks pushed snapshots and commits not already present on a remote for
prohibited metadata, then scans new history for recognized credential patterns.

Completed result directories are immutable. Changes to a method require a new source
snapshot and run identity. Do not regenerate selected cases, alter seeds after inspecting
results, or present development fixtures as held-out evidence.

Repository documentation should describe the software, mathematics, experimental
protocols and reproducible results. Keep personal information, credentials, local machine
paths, assistant instructions and private coordination records outside version control.

See [SECURITY.md](SECURITY.md) for the execution trust model.
