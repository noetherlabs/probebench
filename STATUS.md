# Status

## 2026-10-14 — first development foundation

The supplied destination `noetherlabs/probebench` is private and was empty. Work is on
`codex/probebench-foundation`. The existing noether-ao checkout and scheduler are untouched.
The native task goal is active; an hourly continuation is attached to the same task.
Three initial subagents stopped with workspace-credit errors; do not describe them as
running. Main-task implementation and local tools remain available.

Implemented the six-mode scalar optical reference, bounded OPD commands, noisy camera,
exposure/photon accounting, explicit observation/evaluator separation, four development
policies, source snapshots and integrity-checked retained artifacts. The source-checkout
correctness suite passed **32 tests** using the existing pinned scientific Python.

Retained [development run](results/development-20261014/report.md): 16 episodes, four
fixtures, four methods, three exposures and 600,000 expected photons per episode. Manifest
verification passes. In the paired defocus fixtures, the phase-diversity policy reduced
45 nm WFE to 0.285 and 0.427 nm. Its negative-defocus probe temporarily increased WFE to
125 nm. All trajectories and non-winning methods are retained. The figure was visually
inspected. These are known development fixtures, not held-out benchmark evidence.

No multimodal agent, public release, full controller study, fault-recovery study or twin
qualification study is complete. The least-squares estimator needs noise/convergence
qualification; the initial coordinate search only visits defocus within three exposures.
The model and estimator are perfectly matched, and in-process observation separation
is not a hostile-code sandbox.

The new standalone environment installation is in progress, downloading pinned packages.
Do not claim a fresh-environment reproduction until that process and its tests complete.
Local-only compute remains the spending constraint. Full controller acquisition remains
subject to its unchanged 105,033,365,984-byte storage gate; the user was asked about an
additional local volume. Never delete unrelated files to satisfy that gate.

Next: verify the new environment, review and publish the private development foundation;
then qualify the estimator and define multi-budget, genuinely six-mode comparison runs
before adding the replay explorer and isolated agent interface.
