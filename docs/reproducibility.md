# Reproducing the retained experiments

Each completed experiment includes configuration, numerical runtime versions, a source
snapshot, raw measurements, evaluator records and a SHA-256 file manifest. The dependency
lock is included with the snapshot. Retained datasets are deterministic simulation
fixtures; they are not reserved evaluation cohorts.

## Archived-source reproduction

Run the original ambiguity experiment from its retained source without writing Python
bytecode into the immutable snapshot:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=results/development-20261014/source/src \
  uv run --locked python -m probebench.run --output work/original-reproduction
uv run --locked python scripts/compare_reproduction.py \
  results/development-20261014 work/original-reproduction \
  --output work/reproduction-check.json
```

The recorded fresh-environment reproduction matched all 59 artifact files exactly on
the recorded numerical platform. This is an observed result, not a guarantee of bitwise
identity across different libraries, hardware or operating systems.

The noise comparison can likewise be reproduced from its frozen source:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=results/noise-qualification-20261014/source/src \
  uv run --locked python -m probebench.qualification \
  --config configs/noise_qualification_development.json --output work/noise-reproduction
```

Use the sequential experiment's retained source to reproduce its original estimator
and policy implementation:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=results/sequential-calibration-20261014/source/src \
  uv run --locked python -m probebench.sequential \
  --config configs/sequential_calibration_development.json --output work/sequential-reproduction
```

The sequential archive includes recorded wall-clock durations, which are not expected
to reproduce bit-for-bit. Exact prefix comparisons within the retained experiment and
the resumption check are documented in its [results](sequential-calibration-results.md).
They do not assert fresh-environment reproduction of all sequential artifact bytes.

The matched-estimator schedule ablation has a separate source freeze:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=results/measurement-schedule-20261014/source/src \
  uv run --locked python -m probebench.sequential \
  --config configs/measurement_schedule_development.json --output work/schedule-reproduction
```

Its source includes the explicit unprobed estimator control. Do not substitute a source
version that requires diverse commands for every fit.

## Artifact structure

| Artifact | Contents |
| --- | --- |
| `source/` | Exact Python source, package metadata and dependency lock |
| `configuration.json` or `study.json` | Optical parameters, experiment design, seeds and source/runtime identity |
| `observations/`, `acquisitions/` or `frames/` | Raw signed detector arrays and applied calibration commands |
| `evaluator/` or `cases/` | Latent simulated state, estimated corrections, metrics and fit diagnostics |
| `summary.json` | Complete machine-readable experiment outcomes |
| `report.md` | Tables generated from retained outcomes |
| `manifest.json` | Expected files and their SHA-256 digests |

Evaluation fields and seeds are deliberately included in these development datasets
for reproducibility. They must not be supplied to a controller during an evaluation.

## Verification and resumption

Verification rejects missing files, extra artifacts, altered content and unsafe artifact
paths. Qualification resumption also checks source, configuration and runtime identity,
case records, raw-image integrity and exclusive ownership of the output directory.
Completed cases are not refitted. An interrupted fit reuses its saved acquisition.

The manifest records consistency of an artifact set. It is not a cryptographic signature,
proof of scientific validity, or permission to execute source from an untrusted archive.

The original ambiguity dataset's `final_command_has_camera_observation` field means that
the final command equals the last acquired command. It does not search earlier frames.
Interpret that historical field using its retained source; do not infer an additional
exposure from a submitted command.

The sequential runner records all earlier exposures of the final command in
`final_command_observation_indices`; its `final_command_has_camera_observation` is true
when that list is nonempty. `final_command_is_last_observation` retains the narrower
last-frame condition. Per-exposure checkpoints, completed-fit reuse and verification
of physical metrics are specified in the [sequential protocol](sequential-calibration.md).
