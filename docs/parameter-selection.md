# Global classical-parameter selection

`probebench.selection` selects one setting per classical controller family using a
separate wavefront cohort and a declared global loss. The
[v1 configuration](../configs/classical_selection_v1.json) specifies a coarse parameter
search on the static, monochromatic, six-mode simulated plant. The full acquisition
has not been completed and no selected winner exists for this configuration.

This cohort is public tuning data. Its selection scores are not reserved performance
estimates, and selecting a candidate does not establish superiority or equivalence.

## Declared design

The plant uses 32 pupil pixels per side, oversampling 2, wavelength 550 nm, Gaussian
read noise of 1 electron and a per-mode command limit of 200 nm OPD. The
[wavefront sampler](wavefront-cohorts.md) supplies 12 independent bases: four in each
physical-RMS stratum 10–25, 25–50 and 50–90 nm. Each ordinarily has two conjugate
members; numerical self-conjugates are retained as singletons.

Every candidate receives two detector replicates per member at each combination of
2,000 or 20,000 expected photons per exposure and 3, 7 or 13 exposures. These are six
photon/exposure cells. The v1 selector rejects allowances above 13 exposures, bounding
the observation prefixes retained in fit checkpoints. Increasing the exposure allowance
at fixed photons per exposure also increases total photons. Conjugate members, noise
replicates and nested budget points do not increase the number of independent base wavefronts.

| Family | Candidate parameter, nm OPD | Candidates |
| --- | --- | ---: |
| `no_action_v2` | None | 1 |
| `random_probe_v2` | Random amplitude: 20, 40, 80 | 3 |
| `coordinate_search_v2` | Initial step: 20, 40, 80 | 3 |
| `fixed_diversity_mle_v1` | Defocus probe: 40, 80, 120 | 3 |
| `phase_diversity_mle_v1` | Defocus probe: 40, 80, 120 | 3 |
| `unprobed_mle_v1` | None | 1 |

The 14 candidates use a common policy configuration with only the declared active
parameter overridden. Duplicate effective settings and changes to unrelated candidate
parameters are rejected. Coordinate search keeps a 5-nm step floor; MLE keeps its
existing three starts, tolerances and maximum of 200 iterations per start. This equal
candidate allowance for tunable families is not an equal compute budget.

Coordinate search completes its first six-mode sweep at 13 exposures. The 3- and
7-exposure cells are partial searches; the step floor and step-halving rule are inactive
in this grid. Fixed diversity repeats zero, positive and negative defocus commands.
Adaptive diversity uses camera feedback and can clip a probe about its estimate at the
command bound. The unprobed estimator is an ambiguity control; repeated unprobed
images do not identify the sign of the even modes.

All policies acquire the full allowed exposure count. As in the
[sequential calibration protocol](sequential-calibration.md), the final command may be
submitted without another camera image. Evaluator WFE at that command is not an
additional measurement. Each budget is a separately charged episode, including its
repeated acquisition prefix.

With 24 ordinary members, the design contains 4,032 episodes and 30,912 exposures:

```
episodes  = 14 candidates × 24 members × 2 replicates × 2 photon levels × 3 budgets
exposures = 14 candidates × 24 members × 2 replicates × 2 photon levels × (3 + 7 + 13)
```

These are planned counts, not completed acquisitions. Preparation records the actual
member count and derives the case inventory from it.

## Exclusions and paired streams

The configuration combines an explicit fixture registry with four prior generated
cohort specifications: the 12-base public development example, the 4,096-base shell
sampler test, the 12-base `selection_example` test, and the 12-base development variant
used in a changed-seed resumption test. These describe 4,132 prior generated bases
before any deduplication. The explicit registry also records analytic and protocol-test
fixture vectors.

Preparation regenerates the listed cohorts, checks their declared cluster fingerprints,
and compares the new bases against their full-precision coefficients modulo conjugacy,
using the larger tolerance of each pair of configurations. The v1 tolerance is
$10^{-6}$ nm. An overlap aborts preparation; the sampler does not silently replace a
case. Configuration and regenerated-cohort
digests bind the exclusion evidence. The check covers the supplied inventory, not all
possible optical inputs or an automatically discovered history of experiments.

Detector and policy randomness use separate HMAC-derived domains. Stream identity
includes the protocol, base ID, conjugate member, replicate and photon condition;
it excludes method, candidate and exposure allowance. Candidate order therefore does
not change the draws, random-amplitude candidates share their underlying action stream,
and matching budget prefixes share detector streams. Different commands still produce
different detector distributions and generally different images. Seeds and evaluator
truth remain outside the controller's observation history.

## Global loss and failures

The selection loss is final residual WFE in nanometres, reduced in this order:

1. Average detector replicates within each member.
2. Average members within each independent base.
3. Average bases within each RMS stratum.
4. Average strata within each photon/exposure cell.
5. Average the six cells.

Each stratum has equal weight, each cell has equal weight, and a singleton base has
the same weight as a paired base. The reducer requires identical semantic case coverage
for every candidate and exactly one outcome per expected case. Unknown, duplicate or
missing outcomes and nonfinite losses are rejected.

One minimum-loss candidate is selected per family across all cells. Exact equality of
the unrounded computed losses is resolved by ascending candidate ID, with deterministic
reduction order. There is no tolerance tie, per-cell winner, eligibility filter or
additional failure penalty. The full hierarchical loss table retains every candidate
and its individual outcomes.

Finite nonconverged MLE estimates remain scored at their actual final command and are
counted in `unconverged_fits`. A numerical fit failure uses the existing best measured
camera-score command as fallback. Its residual WFE remains in every average. The
camera score is signed flux in the central 3-by-3 pixels divided by expected photons;
it is distinct from hidden WFE and evaluator Strehl.

`failed_fits` counts estimates marked `numerical_failure`. An episode with any such fit
has `outcome_status="policy_failure"` and `failure_reason="numerical_fit_fallback"`,
even if a later fit succeeds. A finite nonconverged fit alone leaves the episode status
`completed`. Aggregate `failure_count` counts affected episodes, not failed estimates.

An infrastructure interruption, corrupt record or unhandled exception leaves the study
incomplete. Missing cases cannot be dropped or replaced with new draws, and no winner
artifact is produced until the full inventory has been accounted for. Twelve bases
define a reproducible coarse tuning procedure; they do not provide a sample-size
justification for population-level performance claims. Those require a separate
evaluation cohort and prospective cluster-aware analysis.

## Resources and execution

Estimate the complete study before acquisition, from the package root:

```sh
uv run --locked python -m probebench.selection \
  --config configs/classical_selection_v1.json --estimate
```

This validates the configuration and cohort exclusions in memory and reports resource
counts. It acquires no camera observations, performs no estimator fits and creates no
study output directory.

For the 24-member design, 64-by-64 float64 image arrays contribute 1,012,924,416 bytes.
The admission estimate adds 64 KiB per episode for metadata and retained fit diagnostics,
giving 1,277,165,568 bytes. Required free space is

```
operating reserve + 3 × retained-byte estimate + 16 MiB
```

With the configured 1 GiB reserve, a new study requires **4,922,015,744 bytes**. It is
an admission estimate, not measured archive size or a guarantee on all future storage
use. Starting a small chunk still requires admission for the whole study.

On an incomplete resume, already retained evidence is verified under the study lock
before its disk bytes can be credited. Credit is bounded by logical file size, allocated
blocks and canonical serialized payload size, counts each inode once, and excludes the
lock file and unfinished publication buffers. JSON whitespace and ZIP container padding
cannot inflate payload credit. The remaining requirement is the new-study requirement
minus that verified credit, with a floor of the operating reserve
plus 16 MiB. Arbitrary files and unverified byte counts cannot earn credit. This avoids
requiring space twice for verified retained data while preserving the reserve; it does
not reduce the candidate inventory or the study's total retention estimate.

```sh
uv run --locked python -m probebench.selection \
  --config configs/classical_selection_v1.json \
  --output work/classical-selection-v1 --limit-episodes 24
uv run --locked python -m probebench.selection \
  --config configs/classical_selection_v1.json \
  --output work/classical-selection-v1 --resume --limit-episodes 24
```

`--limit-episodes` bounds newly completed episodes in an invocation. Without it,
acquisition continues through the remaining inventory. Repeat the limited resume
command until the full inventory is complete and `manifest.json` has been written.
The two example chunks above do not complete the v1 design. Then verify the archive:

```sh
uv run --locked python -m probebench.selection --verify work/classical-selection-v1
```

`--verify` requires a completed study and is exclusive of generation options.

## Retention and verification

The runner reuses sequential episode acquisition while leaving the existing development
runner and its purpose unchanged. Each frame binds the whole selection study, candidate
ID and effective policy configuration. An exclusive directory lock, complete source
snapshot, numerical runtime and configuration binding protect resumption. Changed source,
runtime or configuration prevents resume; completed archives are verified before any
possible restoration and are not silently repaired.

| Artifact | Contents |
| --- | --- |
| `study.json`, `source/` | Complete configuration, source hashes, runtime and source snapshot |
| `preparation.json` | Selection cohort, exclusion-cohort digests and expected episode count |
| `frames/`, `cases/` | Signed camera arrays, commands, retained fits, evaluator metrics and timings |
| `summary.json` | Every episode's metrics, case identity and failure status |
| `selection.json` | Every candidate's hierarchical loss and one selected ID per family |
| `selected-settings.json` | Full selected policy configurations bound to the study and reduction |
| `manifest.json` | Artifact hashes and final episode/exposure counts |

Verification checks file membership, content hashes and source bindings, regenerates
the cohort and cases, and recomputes derived summaries and winners. Each completed
episode is verified before it can contribute to selection.

Fit records must follow the exact observation-prefix sequence of the declared policy.
For an adaptive-diversity episode with $B$ exposures, the completed record contains one
fit at every prefix length 2 through $B$, in order. Before exposure index $k$, starting
at zero, its checkpoint contains prefixes 2 through $k$; the sequence is empty for
$k<2$. Fixed-diversity and unprobed policies have no acquisition-time fits and one final
fit using all $B$ observations when $B\geq2$.
No-action, random and coordinate policies have no fits. Missing, extra, duplicated or
reordered prefixes are rejected. A numerical-failure estimate must equal the command
with the best measured camera score in that exact observation prefix; fallback cannot
be justified using later frames or hidden WFE. Score ties retain the first observation.

Camera draws and policy commands are replayed using the retained fits, and final
evaluator state is checked. Finite nonconvergence remains explicit. Verification does
not rerun the numerical optimizer or independently establish the optimality of its
saved estimates. Completed episodes are not reacquired during resumption.

Hashes establish internal consistency, not authorship or authenticity. A party able to
replace both an archive and its hashes can change the claimed evidence. Execution and
archived-source reproduction require trusted local code; no process isolation is
provided. See the [artifact trust model](../SECURITY.md) and
[reproducibility documentation](reproducibility.md).
