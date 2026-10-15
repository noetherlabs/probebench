# Wavefront cohorts

`probebench.cohorts` generates reproducible six-mode wavefront cohorts with specified
physical RMS ranges and checks separation from previously exposed wavefronts. It
supports `development` and `selection` splits only. A selection cohort supplies cases;
it does not select a controller. Reserved-cohort generation, reserved outcomes and a
frozen statistical evaluation protocol are not implemented.

## Physical sampling distribution

Coefficients are optical-path differences in nanometres, in this order: defocus,
cosine astigmatism, sine astigmatism, x coma, y coma and spherical aberration. Let
$B$ contain the six piston-removed mode samples over the $N$ illuminated pupil
pixels. The finite-pupil Gram matrix and wavefront RMS are

$$
G=BB^\mathsf{T}/N,\qquad
\operatorname{RMS}(a)=\sqrt{a^\mathsf{T}Ga}.
$$

Each mode is individually normalized in the [optical model](optical-model.md), but
finite sampling need not preserve their mutual orthogonality. The Euclidean norm of
the coefficient vector is therefore not the physical RMS of a general mixture.

For each base wavefront, the sampler draws an RMS $R$ uniformly between the stratum
bounds and an independent standard Gaussian vector $z$. With $G=LL^\mathsf{T}$,
it constructs

$$
q=z/\lVert z\rVert_2,\qquad a=R L^{-\mathsf{T}}q.
$$

Thus $a^\mathsf{T}Ga=R^2$, up to numerical precision, and the direction is isotropic
under the physical RMS metric. Uniform RMS differs from uniform volume in a
six-dimensional shell, which would require a radius proportional to the sixth root
of a uniform volume draw. The implemented distribution is uniform RMS within each
stratum; stratum counts determine the mixture weights.

The specification must satisfy the all-directions correctability bound

$$
R_{\max}\sqrt{\max_j (G^{-1})_{jj}}\leq u_{\max},
$$

where $u_{\max}$ is the per-mode command limit. Generated coefficients are never
clipped. Invalid support, a degenerate direction, nonfinite output, or an overlap
stops generation. There is no silent rejection sampling or replacement draw.

## Conjugate partners and independent bases

For this centrosymmetric pupil, $w(r)$ and $-w(-r)$ have the same in-focus
intensity. In the declared mode order, the partner transform is

$$
C=\operatorname{diag}(-1,-1,-1,+1,+1,-1),\qquad a'=Ca.
$$

The two coma coefficients retain their signs. Negating every coefficient would
implement a different transformation. Each base record contains a `primary` member
and, ordinarily, its `conjugate` partner. If their physical RMS distance is no larger
than `overlap_tolerance_nm`, only the primary member is retained. This handles zero
and numerically self-conjugate wavefronts without duplicating them.

`base_wavefront_id` identifies the independent sampling unit. Its members, repeated
detector draws and nested exposure budgets belong to the same uncertainty cluster;
none creates another independent base wavefront.

## Exclusions and separation

The exclusion registry stores explicit coefficient vectors, unique identifiers and
provenance references using the same optical configuration. Separation is measured
in nanometres, modulo the known conjugate ambiguity:

$$
d(a,b)=\min\left\{\sqrt{(a-b)^\mathsf{T}G(a-b)},
\sqrt{(a-Cb)^\mathsf{T}G(a-Cb)}\right\}.
$$

Generation fails when this distance is at most the configured tolerance, either
against an exclusion or against an earlier base in the cohort. The supplied example
uses $10^{-6}$ nm. The comparison uses full coefficient values, not rounded display
values or hashes. It excludes the specified conjugate orbits; it does not establish
uniqueness against every possible intensity ambiguity or exclude nearby angular
directions at different RMS amplitudes.

The [example registry](../configs/cohort_development.json) covers the three conjugate
orbits exposed by the original and subsequent development fixtures:

| Orbit | Representative OPD coefficients, nm |
| --- | --- |
| Defocus | `[-45, 0, 0, 0, 0, 0]` |
| Mixed A | `[-30, -12, 10, 8, -6, -15]` |
| Mixed B | `[-25, -8, 14, -10, 9, -12]` |

Validation requires exact coefficient coverage of each original fixture or its
conjugate; a nearby vector cannot replace an original exclusion. A complete registry must also
include every subsequently exposed or selected cohort relevant to the intended
split. Recorded source references do not automatically discover that history.
References are descriptive strings: generation does not resolve their pointers or
hash the referenced files. The artifact binds those strings and the explicit vectors,
not the external source bytes. Tests check the supplied example's source pointers.
`assert_disjoint` regenerates and compares supplied cohort configurations, rejects
repeated base identities, and uses the largest supplied overlap tolerance. Its result
applies only to the configurations supplied. Changing a seed or split label alone
does not establish separation.

## Random streams and limits

The sampler identifier is `finite_pupil_isotropic_uniform_rms_pcg64_v1`. A root seed
contains exactly 64 lowercase hexadecimal characters. HMAC-SHA256 derives a 256-bit
seed from the sampler identifier, random domain and canonical JSON identity; NumPy
PCG64 produces the draws. Recognized domains are `wavefront`, `detector`, `policy`
and `bootstrap`. Cohort generation uses only `wavefront`; the other domains are
available for explicit stream assignment by later experiment runners.

Wavefront identities include cohort, split, stratum and base index. Radius and
direction use separate component identities. A base draw therefore does not depend
on stratum order, iteration order or generation chunk boundaries. Records are emitted
in sorted stratum order. Configuration hashes still distinguish reordered stratum
lists, and exact numerical reproduction depends on the recorded numerical
environment. The example seed is public development data, not a secret evaluation
seed.

Each cohort permits at most 4,096 independent bases, and its nonempty exclusion
registry permits at most 4,096 rows. RMS strata must have unique names, positive
integer counts and nonoverlapping intervals; adjacent intervals may share an
endpoint. Pupil sampling is limited to 12–256 pixels per side, with positive integer
oversampling and at most 1,024 focal-plane pixels per side. These are per-cohort
validation bounds, not a guarantee on the aggregate cost of comparing many cohorts.

## Generation, resumption and verification

From the package root:

```sh
uv run --locked python -m probebench.cohorts \
  --config configs/cohort_development.json --output work/cohort-development
uv run --locked python -m probebench.cohorts \
  --config configs/cohort_development.json --output work/cohort-development --resume
uv run --locked python -m probebench.cohorts --verify work/cohort-development
```

The public example specifies four bases in each of three RMS strata: 10–25, 25–50
and 50–90 nm. It therefore requests 12 independent bases and ordinarily 24 members.
These are design counts; the retained artifact records the actual member count after
singleton handling. Generating a cohort acquires no camera exposures and produces no
controller-performance results.

An exclusive output-directory lock protects each run. The study binds the complete
configuration, numerical runtime and source snapshot, including package metadata and
the dependency lock when present. Disk admission preserves the configured operating
reserve. Resumption requires the same configuration, source and runtime and rejects
altered snapshots or a conflicting saved cohort. It does not overwrite a completed
artifact.

`cohort.json` contains the Gram matrix, mode order, configuration digest, base records
and counts. `study.json`, `source/` and `manifest.json` retain the configuration,
runtime, source and file digests. Verification checks artifact membership, paths,
hashes, source bindings and counts, then regenerates the cohort for an exact content
comparison. It uses the executing generator; reproducing an older artifact may
require its trusted archived source and numerical environment.

Hashes detect inconsistent content; they do not authenticate an author or prevent
replacement of both data and hashes. HMAC stream derivation does not authenticate
the archive. This is trusted local execution with no process sandbox. See the
[execution and artifact trust model](../SECURITY.md).

## Requirements for a reserved comparison

A reserved comparison requires disjoint development, selection and evaluation bases,
with complete exclusions and private evaluation seeds and contents. Before evaluation,
the scientific protocol must freeze the sampling population, candidate allowances,
global selection criterion and tie-break, optical and detector models, estimator
settings, resource accounting and failure treatment. Each classical family must have
one setting selected across the declared selection cells; settings must not be chosen
from reserved outcomes.

Primary comparisons should retain paired residual-WFE differences at each declared
photon/exposure condition. Numerical failures, malformed actions and timeouts require
predeclared terminal behavior and complete accounting. Uncertainty must preserve base
clusters, conjugate partners, repeated detector draws and paired methods. Sample size,
multiplicity handling and any equivalence margin require a prospective statistical
specification. Direct-action and tool-assisted agent tracks remain separate. Cohort
generation and a `selection` label alone satisfy none of these evaluation requirements.
