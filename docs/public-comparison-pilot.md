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
and report remain pending; these helpers cannot create reserved streams or keys.
