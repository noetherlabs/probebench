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

Enable the repository's local commit and push checks after installing Gitleaks:

```sh
git config --local core.hooksPath .githooks
```

The commit hook checks the index for prohibited metadata and staged changes for
recognized credential patterns. The push hook checks the outgoing snapshot and every
commit after the destination's advertised tip, including files removed in later commits.
For a new remote branch, it checks the complete reachable history. Commits present on
another remote are still checked. Fetch the destination before pushing if its advertised
tip is not available locally. Local hooks must be enabled in each checkout; they are
not server-side enforcement and do not remove files from previously published history.

Completed result directories are immutable. Changes to a method require a new source
snapshot and run identity. Do not regenerate selected cases, alter seeds after inspecting
results, or present development fixtures as held-out evidence.

Repository documentation should describe the software, mathematics, experimental
protocols and reproducible results. Keep personal information, credentials, local machine
paths, assistant instructions and private coordination records outside version control.

See [SECURITY.md](SECURITY.md) for the execution trust model.
