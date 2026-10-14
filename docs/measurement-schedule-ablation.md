# Measurement schedules and phase ambiguity

This development experiment compares acquisition schedules while retaining the same
six-mode optical model, Poisson–Gaussian likelihood, three optimizer starts, coefficient
bounds and convergence settings. It asks whether a fixed phase-diversity sequence is a
competitive reference for adaptive acquisition, and whether repeated unprobed images
can identify a correction in the presence of a conjugate-phase ambiguity.

## Controls

All methods begin with a zero-command exposure, use the entire allowed exposure budget,
and submit a correction without a final camera exposure. With one image they submit
zero. These one-image points measure abstention, not single-frame phase retrieval.

| Identifier | Acquisition | Final inference |
| --- | --- | --- |
| `unprobed_mle_v1` | Zero command for every exposure. | Fit all images, explicitly allowing nondiverse commands. The chosen minimum is an ambiguous representative. |
| `fixed_diversity_mle_v1` | Repeat the predetermined defocus cycle 0, +80, −80 nm OPD. | Fit all images once after acquisition. |
| `phase_diversity_mle_v1` | Initial zero, then +80 nm defocus; subsequently alternate inferred corrections with signed defocus offsets about the latest estimate. | Fit the available history before each adaptive decision and at submission. |

The adaptive method is defined in the [sequential protocol](sequential-calibration.md).
The first two acquisition commands of fixed and adaptive diversity are identical. At
two exposures, they must therefore retain identical images and return the same final
estimate for a paired case. The unprobed method is an identifiability control, not a
recommended calibration procedure. Disabling the estimator's diversity guard is an
explicit option used only for this control; the default guard remains enabled.

An optimizer's convergence flag does not establish identifiability. For the real,
centred circular pupil, the transformation

\[
T w(\mathbf r)=-w(-\mathbf r)
\]

preserves the zero-command focal-plane intensity. In the implemented basis, it negates
defocus, both astigmatisms and spherical aberration, while preserving both comas. For
any realized detector image, the unprobed likelihood has the same value at these two
wavefronts, up to floating-point error. Repeated exposures reduce noise but cannot
distinguish those hypotheses under this model. Known nonzero even-mode probes can
break this symmetry; the effect is tested using the full propagated intensities.

## Fixed design

`configs/measurement_schedule_development.json` declares four previously published
mixed-mode wavefronts, arranged as two conjugate pairs, two detector replicates,
1/2/3/7 exposures and 2,000/20,000 expected photons per exposure. Read noise is one
electron and the per-mode command bound is 200 nm OPD. The planned design contains
192 episodes and 624 exposures. No wavefront, schedule or optimizer setting is selected
using the resulting outcomes. All fixtures are public development cases, not unseen
combinations or a reserved evaluation cohort.

Detector stream identities are matched across methods and exposure budgets. Different
commands generally yield different camera distributions. Photon conditions reuse stream
identities but are not independent replications. An exposure allowance changes total
photons at fixed photons per exposure, so the study does not isolate the effect of
frame count from total illumination.

The primary descriptive comparison is paired final residual WFE for fixed versus
adaptive diversity within each photon/exposure cell. Report all eight paired cases,
their mean difference, and the number favouring each method; do not attach population
confidence claims to this small related fixture set. Report absolute WFE for every
method, sampled peak Strehl, peak temporary WFE increase, optimizer failures and finite
nonconvergence. For the unprobed control also report residual error modulo the known
conjugacy, `min(RMS(w-u), RMS(Tw-u))`, to distinguish a close ambiguous representative
from a poor fit. This evaluator-only diagnostic does not supply truth to a policy.

Each selected fit has the same three-start numerical allowance. The adaptive method
performs more fits overall; total decision time and fit counts are reported separately.
This is a matched measurement-budget comparison, not a matched computation-budget
comparison. Fit failures retain the measured-score fallback and remain in every table.

## Execution and retention

```sh
uv run --locked python -m probebench.sequential \
  --config configs/measurement_schedule_development.json \
  --output work/measurement-schedule
uv run --locked python -m probebench.sequential --verify work/measurement-schedule
```

The source, configuration and runtime are frozen before acquisition. Every raw image,
command, optimizer start and submitted correction is retained under the existing
[artifact contract](artifact-formats.md). No completed dataset is overwritten. Results
must distinguish measured images from evaluator scores of unmeasured final commands.
This experiment does not evaluate multimodal agents, a mismatched plant, or telescope
performance.
