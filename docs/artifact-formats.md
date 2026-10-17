# Figures, measurement replays and ImageFolder exports

The sequential exporters below accept a completed sequential experiment and verify its manifest,
resource accounting and optical metrics before reading measurements. Outputs are
created in a separate, previously nonexistent directory. The source run is unchanged.
Each output records the input manifest digest and the generating script's digest.
Verification rejects additional episode records or exposure files, including files
listed in a recomputed manifest. Do not modify an input archive while an export runs.

Classical parameter selection has additional candidate, photon-level and base/member
dimensions. Its [reference packet and native archive](../results/classical-selection-v1-20261016/README.md)
use a separate verifier, figure generator, ImageFolder exporter and measurement replay.
The sequential commands below do not support that archive.

## Classical-selection ImageFolder

The release-specific exporter accepts the compact v1 reference packet and the complete
native archive. Follow the reference packet's download and checksum instructions first:

```sh
mkdir -p work
uv run --locked python scripts/export_selection_imagefolder.py \
  results/classical-selection-v1-20261016 \
  --archive downloads/selection-native/classical-selection-v1-20261016 \
  --output work/selection-imagefolder
```

It verifies the full archive through the independently pinned source snapshot before
creating output. A read-only shared lock excludes cooperating archive writers. Each
case and frame is hash-checked again at consumption. Input and output directories must
be disjoint; output must not already exist. Storage admission includes file-allocation
overhead and preserves a 1.25 GiB free-space reserve. The exporter checks remaining
space during writing and publishes its completion manifest only after all files exist.
Interrupted outputs are incomplete and are not resumed or overwritten.

The `public_tuning` split retains every candidate, including nonselected settings, and
every acquired exposure. Image metadata preserves candidate, family, photon allowance,
exposure budget, independent base, conjugate member, RMS stratum and replicate. The
`selected_reference` field identifies the global choice for each family. These cohort
identities and selection flags are analysis labels, not additional controller inputs.

`raw/*.npy` contains the unchanged signed float64 electron arrays. PNGs use the display
transform described below, normalized by each exposure's declared photon level. They
do not replace detector values for inference. No image is synthesized for an unmeasured
final command. Full native case records are preserved under `record` in `episodes.jsonl`,
including latent states, seeds, all optimizer starts and final-command observation
indices. This file is outside the ImageFolder data-file mapping. Columns prefixed
`evaluator_`, complete episode records and configuration state must not be passed to a
controller as observations.

Use the separately pinned Hugging Face environment described below. Streaming loads
avoid building another full image cache:

```python
from datasets import load_dataset
from pathlib import Path
import numpy as np

root = Path("work/selection-imagefolder")
frames = load_dataset("imagefolder",
    data_files={"public_tuning": str(root / "images" / "**")},
    split="public_tuning", streaming=True)
first = next(iter(frames))
electrons = np.load(root / first["raw_array"], allow_pickle=False)
```

The complete v1 export was validated offline with Datasets 5.1.0: all 30,912 PNGs
decoded, every signed array matched the native float64 values, and all 4,032 complete
episode records matched their originals. The metadata retained 13,248 exposures from
selected settings and 17,664 from nonselected candidates. All 55,560,735 negative
detector samples were preserved in the raw arrays.

The export includes a dataset card, configuration, selected settings, rendering and
verification provenance, an independent file manifest, and the existing LICENSE and
NOTICE. It contains public tuning data; it does not introduce a held-out split or a
new scientific result. Local export does not publish a Hugging Face repository.

## Classical-selection measurement replay

The replay generator consumes the verified ImageFolder export above. It copies the
recorded PNGs into a portable directory and embeds the trajectory data in `index.html`;
it does not copy the raw detector arrays or synthesize additional observations.

For an export just produced locally by the verified exporter, capture the manifest
fingerprint and build the replay:

```sh
imagefolder_digest=$(python3 -c 'import hashlib; from pathlib import Path; print(hashlib.sha256(Path("work/selection-imagefolder/manifest.json").read_bytes()).hexdigest())')
uv run --locked python scripts/generate_selection_replay.py \
  work/selection-imagefolder \
  --manifest-sha256 "$imagefolder_digest" \
  --output work/selection-replay
```

For a downloaded export, obtain that fingerprint from a trusted record instead of
computing it from the download itself. The generator checks the supplied fingerprint,
exact inventory, every file hash and the correspondence between episode and exposure
metadata. This establishes consistency with the trusted derived export; it does not
repeat native optical replay or authenticate the source of an arbitrary fingerprint.

Open `work/selection-replay/index.html` in a browser and retain the adjacent `images/`
directory. The replay requires no external services, fonts or scripts. Select any
candidate, photon allowance, exposure budget, base wavefront, conjugate member and
replicate; family and RMS-stratum controls narrow the available choices. Nonselected
candidates remain accessible. Scrub recorded exposures, inspect the evaluator WFE
curve, and zoom or pan the detector image.

The final correction is explicitly classified as measured at the last exposure,
measured earlier, or unmeasured. A recorded-final shortcut is available only when an
actual camera observation exists. Evaluator WFE and Strehl describe latent simulation
state and are not observations supplied to the controller. The display normalization
is fixed within each photon level; raw signed arrays remain in the ImageFolder export.

Input and output directories must be disjoint and output must be new. Admission accounts
for file allocation and preserves the 1.25 GiB operating reserve. Interrupted output
has no completion manifest and must not be treated as a complete replay.

The v1 replay retains all 4,032 episodes and 30,912 exposures. Its final commands comprise
2,016 unmeasured submissions, 1,577 submissions measured earlier, and 439 measured at the
last exposure. Every embedded trajectory and final record was checked against the native
archive, and every copied PNG against the verified ImageFolder. Chromium 145 reached all
4,032 selector combinations and decoded each last exposure plus all 2,016 recorded-final
shortcuts. These browser checks cover endpoints; the generator checks every PNG against
its signed raw array. The replay manifest SHA-256 is
`b0b47b29345e4d42508916c249cdcc06d97f1b0148e0182ad5e9b547d8028f34`.

## Figures and budget-prefix checks

```sh
uv run --locked python scripts/plot_sequential_calibration.py \
  work/sequential-calibration --output work/sequential-figures
uv run --locked python scripts/check_sequential_prefixes.py \
  work/sequential-calibration --output work/sequential-figures/prefix-check.json
uv run --locked python scripts/summarize_sequential_diagnostics.py \
  work/sequential-calibration --output work/sequential-figures/solver-diagnostics.json
```

The figure includes every fixture/noise trajectory; bold lines show descriptive means.
The accompanying JSON records means, measurement status, elapsed decision cost and
selected-fit failures or nonconvergence. No uncertainty interval is inferred from the
small, correlated development cohort. Zero residual errors require a different figure
scale and are rejected explicitly by this logarithmic-WFE figure generator.

The prefix check independently compares retained commands, photon metadata and raw
images at smaller budgets with the matching prefix of the largest budget. Fit timings
are not expected to match between independent executions of the same decisions.

The matched-estimator schedule analysis supports multiple photon levels and retains
every paired fixed/adaptive difference, modulo-conjugacy errors, likelihood symmetry
checks, numerical outcomes and computation counts:

```sh
uv run --locked python scripts/analyze_measurement_schedule.py \
  work/measurement-schedule --output work/measurement-schedule-analysis
```

Its figure uses a numeric logarithmic exposure axis. Horizontal jitter separates paired
case markers at the same budget; it does not represent different exposure counts.

## Offline replay

```sh
uv run --locked python scripts/generate_sequential_replay.py \
  work/sequential-calibration --output work/sequential-replay
```

Open the resulting `index.html` directly in a browser. It contains all images and
metadata, requires no server or external assets, and restricts active content through
a content-security policy. Filters select policy, exposure allowance, wavefront fixture
and noise replicate. Playback, frame selection, zoom and pan operate on recorded
measurements. Final evaluator scores are separately labelled when the submitted
command lacks a camera exposure.

The figure and replay currently accept one declared photon level per exposure. This
matches the supplied sequential development configuration. An export containing
multiple photon levels is refused instead of silently hiding a dimension.

## ImageFolder

```sh
uv run --locked python scripts/export_imagefolder.py \
  work/sequential-calibration --output work/imagefolder
```

The exported layout is:

| File or directory | Contents |
| --- | --- |
| `images/*.png` | Fixed-scale display views of the measured detector arrays |
| `images/metadata.jsonl` | Image paths, episode identities, commands, photon accounting and explicitly named evaluator labels |
| `raw/*.npy` | Signed float64 electron arrays, with object serialization disabled |
| `episodes.jsonl` | Complete evaluator trajectories, latent simulation state, fit diagnostics and final-command measurement status |
| `configuration.json` | Executed experiment configuration |
| `README.md` | Dataset description and explicit `development` split configuration |
| `manifest.json` | Export contents and provenance digests |

`raw_array` paths are relative to the export root. PNGs apply the same fixed display
transform as the replay; they are not substitutes for raw signed data during inference.
Evaluator labels, latent wavefronts and detector seeds support reproducibility and must
not be exposed to a controller during evaluation.

Install the optional, separately pinned validation environment:

```sh
uv venv --python 3.11 work/hf-env
uv pip sync --python work/hf-env/bin/python requirements/huggingface-validation.txt
```

Then load the local export with that environment:

```python
from datasets import load_dataset
from pathlib import Path
import numpy as np

root = Path("work/imagefolder")
frames = load_dataset("imagefolder",
    data_files={"development": str(root / "images" / "**")}, split="development")
electrons = np.load(root / frames[0]["raw_array"], allow_pickle=False)
```

The metadata layout and explicit split mapping follow the official
[ImageFolder](https://huggingface.co/docs/datasets/image_dataset) and
[local-loading](https://huggingface.co/docs/datasets/loading) documentation.
The optional environment is separate from the simulator's numerical environment.
