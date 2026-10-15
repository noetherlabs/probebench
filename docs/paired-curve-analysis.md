# Paired wavefront-error curves

`probebench.paired_statistics.paired_curve` reduces a complete two-method comparison
to final residual wavefront-error (WFE) curves and stratified cluster-bootstrap
intervals. It performs no acquisition or parameter selection. The reference and
target policies must be frozen before evaluating the supplied cohort, with any
[classical-parameter selection](parameter-selection.md) completed on separate data.
This analysis implementation supplies no benchmark result or selected reference.

## API and input schema

```python
from probebench.paired_statistics import paired_curve

result = paired_curve(
    expected_cases,
    outcomes,
    reference_method=reference_method,
    target_method=target_method,
    track=track,
    seed_hex=bootstrap_seed_hex,
    resamples=bootstrap_resamples,
    confidence_level=confidence_level,
)
```

All keyword arguments are required. `reference_method` and `target_method` are
distinct nonempty strings. `track` is one of `classical`, `direct_action` or
`tool_assisted`. It labels the **analysis track**: both arms, including a classical
reference compared with a direct-action or tool-assisted target, carry the requested
track. This label does not verify how a policy operated or which tools it used.

`expected_cases` and `outcomes` are lists of dictionaries. Every dictionary must
contain exactly its schema's fields; extra fields are rejected.

| Expected-case field | Value |
| --- | --- |
| `id` | Unique nonempty case identifier |
| `method` | Requested reference or target identifier |
| `track` | Requested analysis track |
| `stratum_id` | Nonempty RMS-stratum identifier |
| `base_wavefront_id` | Nonempty independent-base identifier |
| `member` | `primary` or `conjugate` |
| `replicate` | Integer at least zero |
| `photons_per_exposure` | Positive integer |
| `exposure_budget` | Positive integer |

| Outcome field | Value |
| --- | --- |
| `case_id` | Identifier of exactly one expected case |
| `residual_wfe_nm` | Nonnegative integer or float convertible to a finite float |
| `outcome_status` | `completed` or `policy_failure` |
| `failure_reason` | `None` for `completed`; nonempty string for `policy_failure` |

Booleans do not satisfy numeric fields. A semantic pair is identified by stratum,
base, photon count, exposure budget, member and replicate. Each pair must contain
both methods exactly once. Every photon/exposure cell must have the same complete
base/member/replicate grid. Each base belongs to one stratum globally and contains
its primary member; a conjugate member is optional. There must be at least two bases
per stratum. Outcome coverage must equal the expected inventory exactly.

The reducer accepts at most 100,000 semantic pairs, hence 200,000 expected rows and
200,000 outcomes, and at most 64 cells. `resamples` is an integer from 2 to 100,000.
The retained matrix is bounded by `resamples × cell_count ≤ 1,000,000`; the resampling
work is additionally bounded by
`resamples × cell_count × independent_base_count ≤ 100,000,000`. These bounds are
checked before curve reduction or bootstrap computation. `confidence_level` must be
finite and strictly between zero and one. `seed_hex` contains exactly 64 lowercase
hexadecimal characters. These validation bounds are not recommendations for a
scientific design. Sample size, resample count and confidence level remain explicit
inputs to a separately declared protocol.

## Estimator

Let $Y^T_{simrc}$ and $Y^R_{simrc}$ denote target and reference final WFE at cell $c$,
stratum $s$, independent base $i$, member $m$ and detector replicate $r$. Differences
are paired before averaging:

$$
d_{sic}=\frac{1}{M_{si}}\sum_m\frac{1}{R_{sim}}\sum_r
\left(Y^T_{simrc}-Y^R_{simrc}\right),\qquad
\widehat{\Delta}_c=\frac{1}{S}\sum_s\frac{1}{n_s}\sum_i d_{sic}.
$$

The recorded criterion is `equal_stratum_base_member_replicate_paired_wfe_v1`.
Negative differences favor the target. The reducer also applies the same hierarchy
to each arm's absolute WFE. Each stratum has equal weight regardless of its base
count, so the estimand is an equal-stratum mixture rather than weighting by stratum
prevalence. A singleton base has the same weight as a base with two members.
Replicate counts may differ between members if declared in the complete inventory;
they do not change member weights. Cells remain separate; there is no overall curve
score or winner selection.

The [cohort construction](wavefront-cohorts.md) defines the independent base and its
conjugate members. Members, detector replicates and exposure-budget points do not
increase the independent-base count. As in the
[sequential measurement protocol](sequential-calibration.md), increasing the exposure
budget at fixed photons per exposure also increases total expected photons.

## Resampling and intervals

For each bootstrap replicate, sample $n_s$ base indices with replacement within
each stratum. Each sampled base contributes its entire vector $d_{si\cdot}$ across
the curve. Average these draws within strata, then average strata equally. The same
base indices are used at every cell. Pairing between arms is already retained in the
base differences. Members, detector replicates, exposures and strata are not
separately resampled, and the reference is not reselected in a bootstrap draw.

Strata, bases and cells are sorted deterministically. Each stratum receives a
separate HMAC-SHA256-derived seed through the cohort `bootstrap` domain, with identity
containing `protocol="paired_wavefront_curve_analysis_v1"`, `stratum_id` and its
sorted `base_wavefront_ids`. NumPy PCG64 generates replacement indices. Input order,
cell order and internal batching do not change the base-index streams. Detector and
policy streams are separate from these bootstrap streams. The result records the
root seed, derived stratum seeds as decimal strings, generator and numerical runtime.

For confidence level $\gamma$, each cell's interval contains the empirical bootstrap
quantiles at $(1-\gamma)/2$ and $(1+\gamma)/2$. The recorded quantile method is
`linear`. For $B$ sorted bootstrap values and probability $p$, the implementation
computes position $h=(B-1)p$ in floating point and interpolates between the values
at $\lfloor h\rfloor$ and $\lceil h\rceil$. Endpoint values and the computed
fractional weight are represented as exact rational numbers for interpolation,
then converted to a float once. This avoids overflow in endpoint subtraction and
premature rounding of weighted endpoints.

Hierarchical means use `math.fsum` followed by division by the number of values.
If summation overflows, the mean is computed with exact rational arithmetic and
converted to a float. Bootstrap draws use the same mean calculation on the original
base contrasts, without scaling or normalization that could erase small terms after
cancellation. Only replacement indices are batched. The result retains every
resampled difference curve, with columns in the same order as `cells`.

Cluster resampling preserves dependence within the sampled units. Its statistical
justification depends on the sampling model and on the number of independent
clusters, not just the number of observations inside each cluster. Field and Welsh
provide context for this distinction; their results do not establish coverage for
this implementation or a particular ProbeBench cohort.
[Field and Welsh, *Bootstrapping clustered data* (2007), primary paper](https://bemlar.ism.ac.jp/zhuang/Refs/Refs/field2007jrssb.pdf).

Intervals are **pointwise**, conditional on the frozen policies and declared
synthetic sampling design. They are not simultaneous curve bands, p-values,
equivalence tests or guarantees of finite-sample coverage. The two-base minimum is
only a computational precondition; small strata can yield unstable uncertainty
estimates. More bootstrap draws cannot compensate for too few independent bases.
`interval_degenerate` is true when the computed endpoints are exactly equal; this
does not establish equivalence or absence of population uncertainty. The intervals
do not include variability from repeating policy training or parameter selection,
and do not establish validity on physical telescopes.

## Failures and verification responsibilities

Finite protocol-defined failure outcomes remain in the estimator and are resampled
with their base clusters. Finite optimizer nonconvergence remains scored at its
actual final command. Under the classical selection protocol, a numerical fit
failure triggers the measured-camera-score fallback and marks its episode as
`policy_failure`, even if a later fit succeeds. The reducer accepts the resulting
status and residual; it does not inspect fit records or reproduce that decision.

`reference_failure_count` and `target_failure_count` count affected episodes, not
individual failed fits. Their denominator is `case_count_per_method` at the same
aggregation level. Counts are summed, independently of the equal-weight mean
hierarchy. Failure flags change these counts but introduce no additional WFE
penalty. Missing, duplicate, malformed or nonfinite outcomes cause rejection; they are
not dropped, imputed or replaced with another wavefront or detector draw.

The acquisition and artifact verifier must establish the physical meaning of the
terminal command, fallback legitimacy, exposure accounting and evaluator WFE. It
must also establish policy/configuration identities, cohort provenance, independence
of bases and separation from development and selection data. The reducer checks the
declared inventory; names and track labels cannot establish those properties.

## Result schema

The returned dictionary has `schema=1` and these fields:

| Fields | Meaning |
| --- | --- |
| `purpose`, `criterion` | `paired_wavefront_curve_analysis_v1` and the estimator identifier above |
| `reference_method`, `target_method`, `track` | Requested comparison identities |
| `contrast`, `units` | `target_minus_reference`, `nm` |
| `interval_scope`, `interval_method` | `pointwise`, `stratified_base_cluster_percentile` |
| `confidence_level`, `quantile_method` | Supplied level and `linear` |
| `independent_base_count`, `stratum_base_counts` | Total bases and a stratum-to-count mapping |
| `input_sha256` | Digest of the validated, canonically sorted input records |
| `runtime` | `python` and `numpy` version strings |
| `cells`, `paired_outcomes`, `bootstrap` | Detailed records below |

Each cell contains `photons_per_exposure`, `exposure_budget`,
`total_expected_photons` (their product), `confidence_interval_nm` (lower, upper),
`interval_degenerate` and `strata`. Cell, stratum and base records all contain
`reference_mean_wfe_nm`, `target_mean_wfe_nm`, `mean_difference_nm`,
`case_count_per_method`, `reference_failure_count` and `target_failure_count`.
Stratum records additionally contain `stratum_id`, `base_count` and `bases`; base
records contain `base_wavefront_id` and `member_count`.

Each `paired_outcomes` record contains `stratum_id`, `base_wavefront_id`,
`photons_per_exposure`, `exposure_budget`, `member`, `replicate`, `difference_nm`,
and `reference` and `target` dictionaries preserving each arm's outcome schema.
`bootstrap` contains `seed_hex`, `resamples`, `bit_generator="PCG64"`,
`stratum_seeds` and `difference_curves_nm`, a resample-by-cell matrix.

`input_sha256` binds the validated records only. It does not include source files,
policy configuration bytes, the choice of reference versus target arm, or bootstrap
arguments, and it does not authenticate their origin. A surrounding artifact must
retain those bindings and the analysis configuration. See the
[artifact trust model](../SECURITY.md). Running this reducer on development or
selection outcomes does not turn them into reserved performance evidence.
