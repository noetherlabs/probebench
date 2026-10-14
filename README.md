# ProbeBench

**Photon-budgeted focal-plane calibration under measurement ambiguity.**

ProbeBench studies how sequential optical measurements resolve ambiguous wavefronts.
A controller observes noisy images of a point source, applies bounded optical-path
corrections, and submits a final command. The evaluator measures residual wavefront
error and focal intensity using latent phase information unavailable to the controller.

For a centrosymmetric pupil, equal positive and negative defocus produce identical
noiseless focal intensities. A known phase offset can distinguish the two states.
This provides a controlled test of measurement selection: an informative probe may
initially degrade the image while enabling a better subsequent correction.

![Measured defocus-diversity trajectories](results/development-20261014/ambiguity.png)

## Optical model

The current implementation uses scalar monochromatic propagation through a circular
pupil at 550 nm, with six controlled modes: defocus, two astigmatisms, two comas, and
spherical aberration. For pupil coordinate $\mathbf r$ and command $\mathbf u_k$,

$$
I_k = \left|\mathcal F\left\{P(\mathbf r)
\exp\left[\frac{2\pi i}{\lambda}
\left(w(\mathbf r)-B(\mathbf r)\mathbf u_k\right)\right]\right\}\right|^2.
$$

Commands and aberrations are expressed in nanometres of optical path difference (OPD).
For an ideal reflective actuator at normal incidence, mirror-surface displacement is
half the applied OPD. The detector combines Poisson photoelectron counts with additive
Gaussian read noise. Every exposure is charged against the expected incident photon
budget. Negative read-noise samples are retained.

The [model specification](docs/optical-model.md) defines discretization, normalization,
units, observation fields and evaluator metrics. The [noise model](docs/noise-likelihood.md)
derives the Poisson–Gaussian likelihood and its numerical approximation.

## Installation

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync --locked --extra dev
uv run --locked python -m pytest -q
```

## Experiments

Run the four-policy optical ambiguity demonstration:

```sh
uv run --locked python -m probebench.run --output work/ambiguity
uv run --locked python -m probebench.run --verify work/ambiguity
```

Compare least-squares and Poisson–Gaussian phase estimation using identical exposures:

```sh
uv run --locked python -m probebench.qualification \
  --config configs/noise_qualification_development.json --output work/noise-comparison
uv run --locked python -m probebench.qualification --verify work/noise-comparison
```

Output directories are immutable after completion. Add `--resume` to an interrupted
qualification run with the same source, configuration and numerical environment.
See [reproducibility](docs/reproducibility.md) for archived-source execution,
artifact schemas and verification scope.

Run sequential calibration with exposure allowances spanning partial and complete
six-mode coordinate sweeps:

```sh
uv run --locked python -m probebench.sequential \
  --config configs/sequential_calibration_development.json --output work/sequential-calibration
uv run --locked python -m probebench.sequential --verify work/sequential-calibration
```

The [sequential protocol](docs/sequential-calibration.md) defines all four policies,
budget pairing, final-command measurement status and exposure-level resumption.
The [measurement-schedule ablation](docs/measurement-schedule-ablation.md) specifies
matched-estimator controls for repeated unprobed and fixed-diversity acquisition.
See [artifact formats](docs/artifact-formats.md) to generate figures, inspect an offline
measurement replay or load a local ImageFolder export with Hugging Face Datasets.

| Experiment | Retained evidence | Purpose |
| --- | --- | --- |
| [Optical ambiguity](results/development-20261014/report.md) | 16 episodes; four policies and four wavefront fixtures | Observation/action accounting and diagnostic probing |
| [Noise qualification](results/noise-qualification-20261014/report.md) | 24 acquisitions; 48 selected estimates across three photon budgets | Estimator and convergence qualification |
| [Sequential calibration](docs/sequential-calibration-results.md) | 192 episodes; 1,632 exposures; six exposure allowances | Complete six-mode search, sequential inference and temporary probe degradation |

## Scope

This is a research implementation with public development fixtures. These fixtures are
not a held-out benchmark cohort. The current experiments assume a static wavefront and
an exactly matched optical model; they do not establish telescope performance, model
mismatch robustness, or multimodal-agent capability. The three-exposure coordinate
baseline in the original ambiguity demonstration probes defocus only. The sequential
runner includes complete six-dimensional sweeps at larger exposure allowances.

The policy API separates observations from evaluator state but does not sandbox
untrusted Python. See [security](SECURITY.md) for execution and artifact trust boundaries.

## Repository structure

- `src/probebench/`: propagation, detector, policies, likelihood and experiment runners.
- `tests/`: independent numerical checks and accounting/integrity regressions.
- `configs/`: explicit experiment configurations.
- `results/`: retained synthetic measurements, evaluations, source snapshots and manifests.
- `scripts/`: reproducibility checks and figure generation.
- `docs/`: scientific specifications and experiment documentation.

See [CONTRIBUTING.md](CONTRIBUTING.md) for validation requirements and
[CHANGELOG.md](CHANGELOG.md) for implementation changes.
