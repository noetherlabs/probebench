# Scalar command-response development study

`probebench.scalar_gain_study` runs one fixed development protocol using the
[nominal-calibration adapter](nominal-calibration.md). It does not define a
held-out benchmark or establish hardware performance. No parameter selection
occurs in this runner.

## Fixed design

| Quantity | Declaration |
| --- | --- |
| Independent bases | 24; eight each in 10–25, 25–50 and 50–90 nm finite-pupil RMS strata |
| Members | Both conjugates of each base |
| Scalar command response | `g = 0.9, 1.0, 1.1`, constant within each episode |
| Classical arms | Frozen `fixed_probe_080` and `adaptive_probe_080` settings |
| Exposure allocation | Seven exposures of 2,000 expected photons; one detector replicate |
| Optics | Six modes, 32-pixel pupil, 2× padding, 550 nm wavelength |
| Detector | Poisson photoelectrons and 1-electron Gaussian read noise |
| Command limit | 200 nm nominal OPD per coefficient; physical response is `g*u` |
| Complete inventory | 288 terminal episodes; at most 2,016 camera frames |

The two arms receive the same nominal calibration and expected-photon allowance.
Their computation differs: at seven exposures the adaptive arm fits prefixes
2–7, while the fixed arm fits once after all observations. Every fit's numerical
status, optimizer status and diagnostics remain in the case record. A finite
camera-based fallback remains scored. Optimizer nonconvergence and a terminal
nonfinite camera observation are distinct outcomes.

## Admission and random streams

Configuration inspection and `--estimate` do not generate wavefronts. Acquisition
requires a separate `FROZEN` receipt binding the complete configuration,
committed executing source, runtime, seed commitments and reviewed exclusion
inventory. The runner checks these bindings and storage admission before drawing
the new cohort. It creates immutable executing-source and reviewed-fixture-source
snapshots. Reviewed repository files must match their committed content at
admission; an arbitrary file-name/hash list is not proof that the fixture audit
was complete. Additional private or external prior evidence must be incorporated
into the reviewed exclusion inventory before its receipt is frozen.

The original 30-row exclusion registry, all four previously declared cohorts,
and the completed classical-selection cohort are required. Additional prior
cohorts remain separate hash-checked inputs; they are not trimmed to fit the
single-registry size limit. Old cohort configurations are reconstructed using
their original optical grids. When grids differ, the same analytic OPD is mapped
modulo piston by

```text
target_coefficient = source_coefficient * raw_mode_RMS(target_grid)
                                        / raw_mode_RMS(source_grid)
```

This uses the fixed six raw Zernike polynomials and unchanged circular aperture.
It is not interpolation of instrument measurements. Cross-grid fixture entries
retain source optics and provenance. Current tests, examples and other exposed
cohorts still require an explicit completeness review; the software does not
discover all historical experiments automatically. An overlapping identity or
physical conjugate orbit aborts preparation. No rejected wavefront is redrawn.

Four distinct 256-bit keys separate wavefront, detector, policy and bootstrap
streams. They are independent inputs, not hashes of source code or outcomes.
The cohort identifier remains `nominal_calibration_development_v1`. Detector and
policy HMAC identities contain exactly `cohort_id`, `stratum_id`, `base_index`,
`member` and `replicate`; gain and arm are omitted for pairing. Case artifact IDs
include gain and arm. Different image distributions can consume random draws
differently, so paired keys do not assert equal realized detector noise.

The bootstrap identity contains `cohort_id`, `analysis="paired_base_curve_v1"`
and `replicates=10000`. One PCG64 stream draws the low, medium and high strata in
that order. Sorting case inputs or changing acquisition chunk boundaries does
not change scientific case identities or random streams.

## Retained artifacts and recovery

Each terminal case is one atomic, uncompressed NPZ containing a complete signed
float64 image stack and bounded JSON metadata. The JSON binds the full case to
the source/configuration/runtime study header, carries public nominal commands,
private physical evaluation, complete saved fits, elapsed episode time and the
raw-image digest. A terminal numerical failure retains its nonfinite final frame,
actual consumed exposures and photons, public/produced inventory counts, and no
invented final metric. Published terminal cases are never retried.

Recovery is at **case publication**, not at each frame. A complete pending archive
is checked and promoted without rerunning the case. Incomplete or corrupt pending
bytes remain untouched and block resumption. If execution stops before any case
archive exists, that unpublished identity can be restarted with the same inputs
and seeds. Consequently retained episode elapsed times do not account for all
work spent in interrupted, unpublished attempts; this runner does not claim
frame-prefix recovery or total interrupted-run compute accounting.

Every case has an enforced 64 KiB JSON allowance plus its bounded raw-array and
container allowance. Source, configuration, preparation, case inventory, reports
and manifest share a separate enforced 16 MiB limit. Admission requires at least
1 GiB operating reserve, three times the retained-byte allowance, and 16 MiB
headroom. Resumption does not reduce that gate. A reserve check also precedes
each new case. These disk checks do not establish a memory or real-time deadline;
runtime readiness must be assessed separately before study admission.

The verifier checks the exact file inventory, hashes and frozen admission before
regenerating the already-bound cohort. It reconstructs commands from retained
fits and nominal camera histories, then replays the physical camera and evaluator
with the declared seeds. This is **retained-path replay**, not reoptimization or
proof that an optimizer found the best solution. Verification uses the recorded
source contents and runtime; it does not silently execute code from an archive.

## Descriptive analysis

For each independent base and gain, final WFE is averaged over its two conjugates.
The primary curve is adaptive-minus-fixed WFE separately at `g=0.9`, `1.0` and
`1.1`, within each stratum and with equal stratum weights. Absolute arm WFE,
degradation relative to unit gain, and difference-in-differences are also reported.
No opposite-gain averaging is a primary estimand.

The 10,000 bootstrap replicates resample independent bases within strata, retaining
each entire gain/arm curve. Frames and conjugates are not independent resampling
units. Reported 2.5/97.5 percentile intervals are pointwise and descriptive; they
do not establish simultaneous coverage, adequate power, equivalence, safety or
an automatic promotion decision. A terminal case without finite final WFE
withholds complete-cohort aggregates. It cannot disappear through a missing-value
mean. All finite fallback outcomes remain included with their statuses.

```sh
python -m probebench.scalar_gain_study --config study-config.json --estimate
python -m probebench.scalar_gain_study --config study-config.json \
  --admission frozen-admission.json --output study-output --limit-episodes 1
python -m probebench.scalar_gain_study --config study-config.json \
  --admission frozen-admission.json --output study-output --resume
python -m probebench.scalar_gain_study --verify study-output
```

Configuration and admission files are study-specific inputs. The completed
[scalar command-response study](scalar-command-response-results.md) retains all
288 episodes, its frozen admission and full verification artifacts. Its public
development cohort must not be reused as reserved evaluation evidence.
