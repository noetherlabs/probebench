# ProbeBench

**Sequential optical calibration under a photon budget.**

Which measurement should an optical controller take next? ProbeBench studies phase ambiguity, noisy focal-plane sensing and the trade between information gained, correction quality and photons consumed.

Each experiment records the exposures, applied phase commands, solver diagnostics and final submitted correction. The simulator models six low-order aberration modes with monochromatic Fourier propagation and Poisson–Gaussian detector noise.

![Known phase diversity resolves a conjugate defocus ambiguity](results/development-20261014/ambiguity.png)

[Optical model](docs/optical-model.md) · [Measurement protocols](docs/sequential-calibration.md) · [Agent interface](docs/agent-protocol.md)

## Experiments

| Study | Evidence |
| --- | --- |
| [Optical ambiguity](results/development-20261014/report.md) | 16 episodes, four acquisition policies |
| [Noise qualification](results/noise-qualification-20261014/report.md) | 24 acquisitions, 48 estimates |
| [Sequential calibration](docs/sequential-calibration-results.md) | 192 episodes, 1,632 exposures |
| [Measurement schedules](docs/measurement-schedule-results.md) | 192 episodes, 624 exposures |
| [Classical parameter selection](docs/parameter-selection-results.md) | 4,032 episodes, 30,912 exposures |
| [Command calibration mismatch](docs/scalar-command-response-results.md) | 288 episodes, 2,016 exposures; paired gain-response curves |

In the classical tuning study, fixed 80 nm phase diversity achieved 1.119 nm aggregate residual WFE, versus 1.347 nm for the tested adaptive schedule. These are development/tuning results on a static, matched optical model; reserved held-out evaluation is still pending.

With ±10% command-response error, adaptive probing reduced pooled WFE but performed worse at low initial aberration. Fixed diversity retained the pooled advantage under correct calibration. The [paired study](docs/scalar-command-response-results.md) reports this reversal with all outcomes and uncertainty intervals.

The [Hugging Face dataset](https://huggingface.co/datasets/noetherlabs/probebench) contains all 30,912 acquired exposures. The `public_tuning` release includes exact signed detector arrays, complete episode records and source manifests. Use the [streaming loader](docs/artifact-formats.md#classical-selection-hub-dataset) to access the complete release.

The separate [scalar command-response export](docs/scalar-command-response-export.md) preserves nominal commands and evaluator-only physical labels for all 2,016 exposures in that study.

## Run

Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/), Linux or macOS.

```bash
git clone https://github.com/noetherlabs/probebench.git
cd probebench
uv sync --locked --extra dev
uv run --locked python -m probebench.run --verify results/development-20261014
uv run --locked python -m probebench.run --output work/ambiguity
uv run --locked python -m probebench.run --verify work/ambiguity
uv run --locked python -m pytest -q
```

Separate runs keep their source, configuration, observations and hashes. See [reproducibility](docs/reproducibility.md) and [artifact formats](docs/artifact-formats.md) for replay and verification.

Use the maintained root environment above. Historical acquisition locks retain vulnerable dependency versions; see [dependency support](SECURITY.md#maintained-dependencies-and-acquisition-snapshots).

## Next questions

Calibration under gain drift, misregistration and defective pixels; epoch-aware calibration updates; measurement selection with information-matrix conditioning; held-out comparisons against the frozen classical baselines.

An evaluator-scored final correction is not a measured camera frame. Current data are simulated, and the local agent protocol is not a benchmark of language-model competence.

Apache-2.0 · [License](LICENSE) · [Contributing](CONTRIBUTING.md)
