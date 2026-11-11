# Prospective comparison contracts

`probebench.comparison_contract` validates a bounded, explicit design for comparing
six frozen classical measurement policies. It performs no cohort generation,
camera simulation, fitting, outcome analysis or file access. Validation does not
register a study, certify exclusions or admit execution. A reserved runner and a
complete-cohort report are not implemented by this module.

The supported design is a **proposal**, not measured held-out evidence. Its 96-base
allocation is a fixed descriptive precision budget, not a power calculation or a
claim that a method will win. The [selected classical settings](parameter-selection-results.md)
remain unchanged. The existing development and selection runners still reject
reserved acquisition.

## Inputs and binding

The API accepts bytes already read by a trusted caller:

```python
from probebench.comparison_contract import validate_comparison_contract

validated = validate_comparison_contract(
    contract_bytes,
    selected_settings_bytes=selected_settings_bytes,
    source_files=source_bytes_by_relative_path,
    exclusion_inventory_bytes=exclusion_inventory_bytes,
)
```

The result contains detached canonical JSON bytes, their SHA-256 and the declared
`proposal` or `synthetic_fixture` status. It is not an execution token. No default
scientific settings are inserted into an incomplete document. Unknown fields,
duplicate JSON keys, nonfinite numbers, oversized input and incompatible choices
raise `ValueError`.

| Contract field | Required content |
| --- | --- |
| `schema`, `purpose` | `probebench.classical-comparison/1`, `classical_heldout_comparison_v1` |
| `status`, `cohort_id` | Proposal or clearly named synthetic fixture; no registered/admitted status |
| `selection` | Exact selected artifact identity, study/selection identities and all six complete policies |
| `population` | Physical sampler, mode order, strata/counts, optics, conjugacy, singleton and overlap rules |
| `experiment` | Photon/exposure cells, repeats, detector, command response, policy information and stream pairing |
| `analysis` | Primary/secondary roles, hierarchy, multiplicity, resampling, interpretation and stopping rules |
| `failures` | Retained fit failures, incomplete evidence, interruption, resumption and cost accounting |
| `sources` | Revision, exact role-to-path map and hashes of all supplied source bytes |
| `runtime` | Declared dependency pins, host labels and interpreter digest |
| `exclusions` | Exact inventory-byte digest and pending full-precision verification state |
| `custody_schema` | The separate future custody-statement schema |

The selected-settings raw digest is pinned to
`c858028008125b277bb6f9dcf1696e9ada1e0b5bada1ab19a577688cb0e50cab`.
A caller cannot substitute altered settings together with a new claimed digest.
The six selected optical/policy/estimator source files also retain their original
hashes. Other supplied sources are content-bound with mandatory semantic roles.
Source manifests accept normalized relative paths only and must match the
supplied byte inventory exactly. This checks the supplied snapshot, not whether
it is a complete live checkout or the code later executed.

Runtime fields describe a proposed runtime; validation does not import numerical
libraries or inspect the host. Inventory content is hash-bound, but this API does
not reconstruct historical wavefronts, discover missing exposures, map physical
bases or establish disjointness. Those remain mandatory pre-acquisition checks.
Hashes detect inconsistent bytes; they do not authenticate their origin or prove
that a reserve has remained unseen.

JSON input is limited to 1 MiB, depth 16 and 20,000 values. In-memory canonical
encoding first checks a conservative size bound, including escaped Unicode, so
it may reject a document whose eventual encoding would be smaller than 1 MiB.
Integers are bounded to 63 magnitude bits. Source input is limited to 128 files,
1 MiB each and 16 MiB in total. These parsing limits are not resource admission
for an experiment.

## Supported design

The six families retain one selected setting each: no action, unprobed MLE,
20 nm random probing, coordinate search starting at 40 nm, fixed 80 nm diversity
and adaptive 80 nm diversity. Inactive policy fields, estimator starts and
tolerances remain bound too. No settings are selected from evaluation outcomes.

The proposal has 32 independent bases in each of three finite-pupil RMS strata:
10–25, 25–50 and 50–90 nm. Both conjugate members are used except a sampler-defined
singleton, which retains full base weight. Each member has two detector repeats.
The six cells combine 2,000/20,000 photons per exposure with 3/7/13 exposures,
including the initial image. The optical model uses a 32-pixel circular pupil,
2× Fourier oversampling, 550 nm wavelength, six OPD modes and 1-electron Gaussian
read noise alongside Poisson photoelectrons. Actual and nominal command gain
are one. Final submission is scored by the evaluator without a new exposure.

The primary family contains **six adaptive-minus-fixed final-WFE contrasts**.
Pair before averaging repeats, members, bases and then strata equally. Resample
whole base curves within strata with 50,000 bootstrap draws and linear quantiles.
Marginal confidence is `1 - 0.05/6`; Bonferroni adjustment targets nominal 95%
familywise coverage **conditional on adequate marginal bootstrap coverage**.
It does not make percentile intervals exact. A cell advantage requires its
primary upper endpoint strictly below zero. An across-curve advantage requires
all six cells; no outcome-dependent choice of favorable cells or stopping.

The other four methods versus fixed diversity are descriptive comparisons with
10,000 draws and pointwise 95% intervals. They cannot be promoted into the primary
family after inspection. There is no equivalence or noninferiority claim.
Conjugates, repeats, methods and nested budgets do not increase the independent
base count. The sample-size label explicitly disclaims a powered design.

Finite nonconvergence remains scored; the existing numerical-fit failure uses
its measured-camera-score fallback. Unexpected failures or nonfinite evaluator
outcomes withhold complete-cohort contrasts. Preserve all raw evidence and
interrupted attempts; never replace a failed base or invisibly retry an unknown
dispatch. Logical photons/exposures and replay/optimizer computation are separate
costs. Equal photon budgets do not imply equal computation or a causal test of
adaptivity alone. This static population does not establish hardware or on-sky
performance.

## Separate custody statements

Actual keys and truth-bearing cases are absent from the scientific contract.
After a future immutable scientific freeze, a separate custody statement can bind
four domain commitments to its canonical hash:

```python
from probebench.comparison_contract import validate_custody_receipt

binding = validate_custody_receipt(
    custody_bytes,
    contract=validated,
    selected_settings_bytes=selected_settings_bytes,
    source_files=source_bytes_by_relative_path,
    exclusion_inventory_bytes=exclusion_inventory_bytes,
)
```

The exact fields are `schema`, `status`, `scientific_contract_sha256`,
`commitment_scheme` and `commitments`. Schema is
`probebench.comparison-custody/1`; the scheme identifier is
`sha256_domain_separated_256bit_keys_v1`. The commitments map contains distinct
SHA-256 strings for `wavefront`, `detector`, `policy` and `bootstrap`.
Synthetic fixtures require a `synthetic_fixture` statement; proposals accept
only an `unverified_statement`.

The supplied evidence is revalidated, and the object's canonical bytes, identity
and status must match the newly validated contract. Constructing or replacing
fields of the public value object does not bypass those checks. This still does
not authenticate caller-supplied evidence or prove an earlier validation occurred.

The returned scientific and custody identities remain separate. Binding does
not verify key existence, entropy, secrecy or creation time, and explicitly
returns `execution_admitted=False`. No key-generation or commitment-production
function is supplied. Tests use invented commitments and make no scientific
preregistration claim. A future acquisition gate must independently establish
source/runtime, exposure-history closure, physical separation, resources and
the retained chronology before allowing any reserved generation.

The [public-fixture pilot and exposure batches](public-comparison-pilot.md)
support development qualification and source-bound historical checks. They do
not admit reserved execution or establish complete historical exclusion closure.

The [native comparison layer](native-comparison.md) implements complete-inventory
checks, retained case execution and report reduction. Its acquisition entry is
restricted to public development fixtures; prospective execution remains pending.
