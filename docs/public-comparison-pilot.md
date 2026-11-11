# Public-fixture qualification

`probebench.comparison_pilot` exercises the six frozen classical policies on
the existing `mixed_a` development fixture. Each policy receives 13 exposures
of 2,000 expected photons, including the initial image, on the 32-pixel pupil
at 550 nm with 2× Fourier oversampling and unit command gain. It reuses the
public detector and policy seeds from
[`measurement_schedule_development.json`](../configs/measurement_schedule_development.json).
This is execution and resource qualification on exposed material. It provides
no independent held-out samples, confidence intervals or policy-superiority claim.

The exact selected-settings bytes and the six selected optical/policy source
digests are checked before optics are constructed. Native study admission binds
the current complete source inventory, configuration and runtime. Run only under
an external duration, process-group memory and storage supervisor. Supply its
operating reserve explicitly; the runner refuses a reserve below 1 GiB.

```bash
python -m probebench.comparison_pilot /path/to/fresh-pilot \
  --selected-settings results/classical-selection-v1-20261016/selected-settings.json \
  --operating-reserve-bytes 10737418240

python -m probebench.comparison_pilot /path/to/fresh-pilot --verify
```

The reserve in this example is 10 GiB, not an estimate of total experiment cost.
The runner adds retained-output headroom; another active workload may require
a larger reserve. A complete packet contains six episodes and 78 camera
observations. Runtime and disk measurements must come from an actual supervised
execution; the interface and injected tests are not such measurements.

Before each policy or camera callback, the runner publishes an exclusive intent.
Commands, images, fit records, evaluator data and terminal records are retained
separately. Policies receive only the public calibration, instrument bounds,
selected settings, policy seed and immutable nominal observation history. The
driver alone holds the incident vector, detector seed and actual command gain.
This is an information boundary for trusted built-in policies, not an operating
system sandbox against malicious code.

Finite nonconvergence and the existing camera-score fallback remain scored.
Unexpected exceptions, invalid commands and nonfinite observations stop the
pilot and withhold its completion manifest. Produced raw arrays, including
nonfinite pixels, are retained when the writer remains available. Nonfinite
diagnostic scalars use explicit `nonfinite_float` tags. An I/O failure or hard
process termination can leave partial evidence and an intent with unknown
outcome; the original directory is preserved and **cannot be resumed or adopted**.
No callback is automatically retried. A fresh diagnostic run must be identified
separately and must not replace the stopped evidence.

The read-only verifier checks the exact artifact inventory, source bindings,
command/observation chronology, array headers, hashes and counters. It never
constructs optics, draws noise or refits a policy. This is retained-artifact
validation, not independent recomputation of the optical model or optimizer.
It first admits a stable packet of regular files: at most 2,048 entries, 1 MiB
per file and 128 MiB in total, before native hash verification. Symlinks and
unreadable directories are refused. Unit-gain physical commands, mirror-surface
units, trajectory counters and final-command observation flags must agree with
the retained nominal commands; public metadata is reconstructed exactly.
Camera-call attempts, completed public observations, retained fits and wall time
remain distinct; the packet does not claim to count every internal likelihood
evaluation or Fourier transform.

## Exposure batches

`probebench.exposure_registry` resolves already-public wavefront occurrences
from content-bound JSON sources. A batch manifest has these fields:

```json
{
  "schema": "probebench.exposure-batch/1",
  "purpose": "already_exposed_public_material",
  "sources": {"fixture": "<sha256 of exact source bytes>"},
  "occurrences": [{
    "occurrence_id": "fixture-0",
    "base_id": "mixed_a",
    "vector_source": "fixture",
    "vector_pointer": "/wavefronts/0",
    "optics_source": "fixture",
    "optics_pointer": "/optics"
  }]
}
```

The source at `/wavefronts/0` must contain six finite OPD coefficients in nm;
`/optics` must contain `pupil_pixels`, `oversampling` and `wavelength_nm`.
Sources can be separate for the vector and its optical normalization. The digest
placeholder above is illustrative and is rejected as an actual manifest.

```python
from probebench.exposure_registry import compile_exposure_batch, check_exposure

batch = compile_exposure_batch(manifest_bytes, source_files=public_json_by_name)
check = check_exposure(
    base_id, coefficients_nm, target_optics=target_optics,
    manifest_bytes=manifest_bytes, source_files=public_json_by_name,
)
```

An occurrence identifies its source location. Reusing a base ID with different
coefficients retains every occurrence. The check reports ID collisions separately
from physical overlap: it maps the raw analytic OPD polynomial between finite
pupil normalizations, removes piston and measures RMS distance to both `w(r)`
and `-w(-r)`. It makes no claim that these are every possible intensity ambiguity.
Coefficients are never rounded, clipped or substituted. The minimum overlap
tolerance is 1e-6 nm.

Each batch allows at most 256 occurrences and 64 bounded JSON documents, each
at most 1 MiB and collectively at most 16 MiB. The common JSON structural bounds
also apply. Optical grids are limited to 256 pupil and 1,024 focal pixels;
coefficients are limited to ±1e6 nm and wavelength to (0,1e9] nm. Larger histories
require separate content-bound batches. Missing, malformed, mismatched or
unsupported source material remains explicitly unresolved, and an unresolved
batch cannot return a clear-overlap result. These are consistency checks on
supplied evidence, not authentication of its claimed provenance.

Neither a resolved batch nor a clear check establishes complete historical
exclusion closure. All historical occurrences, source normalization, aliases and
new development exposures must be accounted for before a prospective cohort is
admitted. The [reserved comparison](reserved-comparison.md), its complete runner
and its prospective results remain pending; these helpers cannot create reserved
streams or keys. The [native comparison runner and report](native-comparison.md)
currently admit public development execution only.

## Historical cohorts and complete supplied histories

`probebench.exposure_history` joins the bounded batches above. This is the next
input to historical exclusion review, not a declaration that the full past
exposure history is known or that a future cohort is independent.

For a previously published development or selection cohort, prefer the original
full-precision cluster records and independently retained fingerprints:

```python
from probebench.exposure_history import package_public_cohort

packaged = package_public_cohort(
    public_configuration_bytes,
    original_clusters_json_bytes,
    history_id="published_cohort",
    expected_configuration_sha256=retained_configuration_fingerprint,
    expected_clusters_sha256=retained_clusters_fingerprint,
)
```

`original_clusters_json_bytes` is the retained JSON cluster array. The helper
checks both canonical fingerprints before packaging; it constructs no optics and
invokes no sampler. Each input retains the existing 1 MiB JSON byte and structural
bounds. Larger retained histories use the exposure-batch interface directly.
The original configuration bytes and supplied cluster bytes receive separate
byte hashes; all cluster values are preserved when partitioned into batches.
Declared stratum counts and every cohort/stratum/index identity must match.
Independent provenance must also establish that these records used this optical
configuration: matching labels alone cannot authenticate physical normalization.

When the original records are unavailable, `reconstruct_public_cohort` accepts the
same arguments except `original_clusters_json_bytes` and invokes the existing
sampler. It returns batches **only** if the regenerated cluster fingerprint matches
exactly. Source, seeds and package versions alone do not ensure bitwise replay:
floating-point linear algebra can differ across platforms and numerical libraries.
A changed final bit is refused, even if the difference is physically negligible.
The error reports the expected and actual hashes. Do not replace a historical
fingerprint with the current machine's hash or round coefficients to make it pass.
Use the original records or reproduce the original numerical environment; until
then that historical source remains unresolved.

Reserved splits and changed configuration fingerprints are refused before
sampling. Neither helper draws a new key, simulates a camera or writes a file.
Reconstruction is numerical work and needs external duration, memory and storage
supervision. Caller-supplied hashes are not authentication or proof of earlier
publication. Portable tests package retained records and check strict rejection;
a passing runtime-contract test does not mean a historical replay succeeded.

Each primary base occurrence retains its identity, full cluster record and source
pupil normalization. Checking its direct and conjugate orbit also excludes its
partner. No coefficient rounding, approximate replacement or identity-based
deduplication occurs. Large cohorts are split into batches of at most 256
occurrences; no per-batch source bound is raised.

Before checking a candidate, independently declare **all** required batches:

```json
{
  "schema": "probebench.exposure-history/1",
  "purpose": "declared_already_exposed_public_material",
  "batches": [{
    "batch_id": "published_cohort:0000",
    "manifest_sha256": "<exact exposure-batch manifest SHA-256>",
    "occurrence_count": 12,
    "overlap_tolerance_nm": 0.000001
  }]
}
```

The digest placeholder is illustrative and is rejected by the API. The returned
`inventory_bytes` from either helper covers that one cohort only. It must not be
mistaken for the independently reviewed whole history. Include fixture and alias
batches, other cohorts, and known occurrences from failed or partial prior runs.
A base ID alone cannot supply missing coefficients or physical normalization.

```python
from probebench.exposure_history import compile_exposure_history, check_exposure_history

resolved = compile_exposure_history(history_inventory_bytes, batches=all_batches)
checked = check_exposure_history(
    candidate_base_id, candidate_opd_nm,
    target_optics=target_optics,
    inventory_bytes=history_inventory_bytes,
    batches=all_batches,
    tolerance_nm=0.000001,
)
```

`all_batches` maps each declared batch ID to `manifest_bytes` and `source_files`,
using the same byte inputs as the single-batch API. The join checks exact batch
hashes and occurrence counts. Missing or mismatched batches and unresolved sources
cannot reach physical checking or return clearance. Duplicate occurrence IDs or
duplicate same-identity source locations are refused across batches. Legitimate
aliases at distinct occurrences and different historical vectors sharing an ID
remain separate; neither is silently discarded.

Physical checks reuse the existing finite-pupil, conjugate-aware calculation at
the maximum of the requested and each batch's declared historical tolerance.
The history owner must preserve the applicable tolerance from each source.
Identity collisions and physical overlaps retain their batch and occurrence IDs.
`clear_of_declared_history` is conditional on the supplied evidence. Both
`historical_closure_established` and `execution_admitted` remain false.

The join permits at most 64 batches and 16,384 declared occurrences. It admits
at most 64 MiB of manifest/source inputs in aggregate, conservatively counting
repeated source bytes; individual batch and JSON bounds still apply. It performs
no filesystem discovery. Complete supplied coverage does not establish that an
unrecorded experiment never occurred, that a reserve remained unseen, or that
source/runtime/resource/custody admission has been completed.
