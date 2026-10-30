# Scalar command-response measurement export

`scripts/export_scalar_gain_imagefolder.py` converts the complete
[scalar command-response release](../results/scalar-command-response-v1-20261028/README.md)
to a local ImageFolder dataset. It accepts only the released raw manifest
`ee56441561fa50aeb80f1e33e516cf9151248445f75d997e0d46e9b5002c3322`:
288 episodes and 2,016 acquired images from 24 independent base wavefronts.
The input is the extracted `scalar-command-response-v1/` directory, not the
compact results packet. Supply it as the positional argument and choose a new,
disjoint output directory with `--output`.

The exporter checks the exact archive inventory, every retained file hash,
configuration/source bindings, complete base/member/gain/schedule grid, bounded
array headers, nominal observation schema and photon accounting. It holds the
archive's shared lock and rechecks consumed case bytes and the final inventory.
Archived Python sources are hashed but never executed. This is export consistency
verification; it does not rerun optical simulation, optimization or the separate
numerical audit.

| Artifact or field | Meaning |
| --- | --- |
| `images/*.png` | Fixed-scale display rendering of one retained exposure. |
| `images/metadata.jsonl` | One row per acquired exposure; contains the fields below. |
| `observation_schema` | `probebench.nominal-observation/1`. |
| `commanded_opd_nm` | Requested six-mode command in nominal OPD nanometres. |
| `nominal_applied_opd_nm` | Unit-response calibration prediction; not physical actuation telemetry. |
| `nominal_calibration` | Identity response, nominal-command units and predicted OPD units. |
| `expected_photons`, `cumulative_expected_photons` | Consumed acquisition budget. |
| `evaluator_command_gain`, `evaluator_applied_opd_nm` | Simulator-only true gain and physical command. |
| `evaluator_wfe_nm`, `evaluator_strehl_peak_ratio` | Retained simulator scores at the acquired command. |
| `raw_array`, `raw_array_sha256` | Export-relative signed float64 NPY and whole-file SHA-256. |
| `image_electrons_sha256` | SHA-256 of the unchanged C-order float64 sample bytes. |
| `source_case_sha256` | SHA-256 of the native episode NPZ containing the frame. |
| `episodes.jsonl` | Complete native records, including latent state, seeds, fits and final-command status. |
| `source-manifest.json`, `provenance.json`, `manifest.json` | Native release binding, conversion details and complete derived-file inventory. |

There is no unqualified `applied_opd_nm` observation column. Relabeling a nominal
command as known physical actuation would expose an assumption contradicted by
the experiment. The older sequential/selection exporters and replay loaders do
not support this format. Full episode records, configuration and evaluator fields
are reproduction material, not controller observations. Cohort identities and
truth-derived RMS strata are also analysis labels.

The generated dataset card uses configuration `scalar_command_response_v1`, split
`development`, and an explicit `images/**` data-file mapping. Future Hub packaging
must preserve this separate configuration; it must not append rows to the existing
classical-selection `default/public_tuning` schema or move that release's tag.
The exporter itself performs no upload or remote mutation.

For local streaming, use `load_dataset("imagefolder", data_files={"development":
str(root / "images" / "**")}, split="development", streaming=True)` in the separately
pinned Hugging Face validation environment. After verifying the export manifest
against a trusted digest, reconstruct an observation explicitly:

```python
import hashlib
import io
import numpy as np
from probebench.nominal_calibration import NominalObservation

assert row["observation_schema"] == "probebench.nominal-observation/1"
payload = (root / row["raw_array"]).read_bytes()
assert hashlib.sha256(payload).hexdigest() == row["raw_array_sha256"]
electrons = np.load(io.BytesIO(payload), allow_pickle=False)
assert hashlib.sha256(electrons.tobytes()).hexdigest() == row["image_electrons_sha256"]
observation = NominalObservation(
    index=row["exposure_index"],
    commanded_opd_nm=tuple(row["commanded_opd_nm"]),
    expected_photons=row["expected_photons"],
    cumulative_expected_photons=row["cumulative_expected_photons"],
    image_electrons=electrons,
)
```

Only the explicit observation projection belongs at a controller boundary. The
PNG is display-only: negative samples are clipped for rendering and retained
unchanged in the signed array. Every frame corresponds to an acquired exposure;
no image is generated for an unmeasured final submission. Conjugates and gains
share base identities; 2,016 frames are not 2,016 independent wavefronts.

Storage admission accounts for filesystem allocation and preserves a 1.25 GiB
reserve. An interrupted output is retained without a complete manifest and is
neither resumed nor overwritten. Source data remain unchanged. Hashes establish
consistency with the pinned release, not authenticity against a malicious writer.
