# Scalar command-response study: retained evidence

This packet accompanies [Calibration mismatch changes the preferred measurement
schedule](../../docs/scalar-command-response-results.md). It contains 24 independent
bases, two conjugate members, three command gains and two frozen classical schedules:
288 episodes, 2,016 exposures and 4,032,000 expected photons.

The full [raw archive](https://github.com/noetherlabs/probebench/releases/download/scalar-command-response-v1-20261028/scalar-command-response-v1-20261028.tar.gz)
includes top-level `LICENSE` and `NOTICE` files alongside the study directory.
The study contains all signed float64 detector arrays, fitted paths, physical evaluations,
source snapshots, admission and inventory. This compact Git packet retains exact
copies of the study header, preparation, summary, analysis and raw manifest, plus
figures and the independent audit receipt. It is not itself a full replay input.

| Artifact | SHA-256 |
| --- | --- |
| Raw archive, 64,958,900 bytes | `3c08df9e04c4b5110893ba758f13f251c36f59b6f6a4372c224d2045371b7e38` |
| Raw manifest | `ee56441561fa50aeb80f1e33e516cf9151248445f75d997e0d46e9b5002c3322` |
| Frozen canonical configuration | `47145e03b9c96bc75256cd56d27ed55400ee5131c525fe57410fd76eaad5c5e8` |

Acquisition source: [`99e5e883c9e064fdf778d8cbea7e439e7f09e91b`](https://github.com/noetherlabs/probebench/commit/99e5e883c9e064fdf778d8cbea7e439e7f09e91b).
Recorded runtime: Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, macOS 14.6 arm64.
No parameters were selected from this cohort.

## Independent numerical audit

Download the archive and `SHA256SUMS` from the [release](https://github.com/noetherlabs/probebench/releases/tag/scalar-command-response-v1-20261028),
verify the checksum, and extract it into a fresh directory. The resulting
`scalar-command-response-v1/` directory is the full study input. From this
repository, run:

```sh
python scripts/audit_scalar_gain.py --run scalar-command-response-v1 \
  --out independent-audit.json
```

The audit is specific to this release and pins its complete manifest. It requires
NumPy 2.4.6 for the declared random stream and otherwise uses the Python standard
library. It checks the complete
declared grid, retained hashes, physical WFE, fitted command prefixes and paired
reductions. Its implementation is independent of the ProbeBench package. Its
receipt records the source script digest and observed numerical differences.
It does not independently regenerate camera noise, Strehl, historical exclusions
or optimizer solutions. See the [methods](../../docs/scalar-gain-study.md) for
the distinction between replay and reoptimization.

## Full retained-path replay

Use the acquisition source revision and its recorded runtime with the full study
directory, then run:

```sh
python -m probebench.scalar_gain_study --verify scalar-command-response-v1
```

This verifier enforces exact source/runtime bindings. A different runtime can
inspect the evidence and run the separate numerical audit, but does not receive
an exact-replay verdict by bypassing those checks. No archived Python code is
automatically executed. The final submitted WFE is simulator evaluation, not a
newly acquired corrected image; episode times are not a latency benchmark.

## Figures and compact inventory

```sh
python scripts/plot_scalar_gain.py \
  --analysis results/scalar-command-response-v1-20261028/analysis.json \
  --out scalar-command-response-figures
```

The plot reads retained intervals directly. `figures/manifest.json` binds its input
and outputs. `packet-manifest.json` hashes this compact packet; `raw-manifest.json`
is the unchanged manifest of the separate full study. All scientific data here
are public development evidence. Pointwise bootstrap intervals do not provide
simultaneous coverage, a held-out result or instrument qualification.
