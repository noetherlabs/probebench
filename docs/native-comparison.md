# Native classical comparison

The comparison layer records the six selected classical policies under the
existing sensing budgets. It separates case completeness, artifact consistency,
optimizer status and statistical outcomes.

The acquisition command currently accepts **only the public `mixed_a` development
fixture**. Prospective execution remains closed pending full-precision historical
exclusion closure and separate custody/resource admission. The proposed
[reserved comparison](reserved-comparison.md) has not been run by this command.

## Development execution

```bash
uv run python -m probebench.comparison /path/to/new-packet \
  --selected-settings results/classical-selection-v1-20261016/selected-settings.json \
  --operating-reserve-bytes 1073741824 --limit-cases 6

uv run python -m probebench.comparison /path/to/new-packet \
  --selected-settings results/classical-selection-v1-20261016/selected-settings.json \
  --operating-reserve-bytes 1073741824 --resume

uv run python -m probebench.comparison /path/to/new-packet --verify
uv run python -m probebench.comparison_report /path/to/new-packet /path/to/new-report
```

This evaluates 36 cases: six methods, two photon levels (2,000 and 20,000 per
frame), and three exposure budgets (3, 7 and 13). Each case reuses the exposed
`mixed_a` wavefront and existing public detector/policy streams. There is one
replicate per cell. These cases provide no population confidence interval and
must not be relabelled held-out evidence.

The initial zero-command exposure consumes budget. Final submission acquires no
new exposure. Final WFE and Strehl are evaluator computations; a previously
observed identical command is not a fresh post-decision confirmation. Reports
retain the acquired-frame WFE sequence, maximum probe-induced increase, static
command-bound check, camera-observation flag and fit/optimizer accounting.
Equal sensing budgets do not imply equal computation.

`--limit-cases` stops at a sealed case boundary. Resume verifies the complete
existing acquisition prefix before constructing optics. A completed case is never
re-executed. A dispatched case without a complete binding, an unexpected error or
a stopped prefix requires explicit reconciliation; resume does not retry it.
Raw commands, frames, fit histories, dispatches and failure records remain on disk.
The native reserve check does not enforce wall time or process memory; these
require separate execution supervision.

## Complete inventory and analysis

`comparison_inventory` validates the proposed 96-base design before extracting
any two policy arms: exactly 32 base identities in each of three strata, each
base's declared primary/conjugate membership, two detector repetitions, six
methods and all six sensing cells. A physically established singleton retains
one base's weight. Metadata validation alone proves neither physical independence
nor exclusion from previously exposed wavefronts.

`comparison_report.analyze_complete` revalidates the scientific contract, selected
bytes and supplied source evidence. It requires the complete independently
declared case inventory before invoking the paired reducer. Missing outcomes or
retained unexpected failures withhold complete contrasts; they receive no
invented finite penalty. Finite optimizer nonconvergence and the existing
camera-score fallback remain scored, with their diagnostics retained separately.

The primary adaptive-minus-fixed analysis is one six-cell call with 50,000
whole-base bootstrap draws and marginal confidence `1 - 0.05 / 6`. Its familywise
interpretation is nominal and conditional on marginal bootstrap coverage. Four
secondary method-minus-fixed analyses use 10,000 draws and pointwise 95%
intervals. They are descriptive. Repetitions, conjugate members, bases and strata
are averaged in that order; entire base curves are resampled together.
An interval containing zero does not establish equivalence. Degenerate intervals
remain flagged. The fixed 96-base budget is not a powered novelty claim.

The analysis envelope binds the full arguments, source/runtime, contract,
inventory, case-result shards, raw manifest and custody statement. Supplied hashes
are content bindings, not authentication or execution authorization. The pure
analysis API cannot establish held-out provenance. A public-fixture inventory is
refused at the population-analysis boundary.

## Artifact layout

A comparison has one manifest per case and one per base. The root manifest binds
all base manifests and the retained source/runtime header. Exact per-case schemas
and command/observation consistency are checked independently of hash agreement.
Verification reads recorded arrays and metrics; it does not rerun optics,
reconstruct detector noise or refit policies.

Analysis output stores per-case summaries, paired outcomes and bootstrap draws in
bounded JSON shards. It includes private bootstrap seed material and requires a
separate publication review. Changing filenames or hashes cannot turn synthetic
fixtures into a scientific preregistration, resolve historical provenance, or
admit a new cohort.
