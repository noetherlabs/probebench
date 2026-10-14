# Figures, measurement replays and ImageFolder exports

All exporters accept a completed sequential experiment and verify its manifest,
resource accounting and optical metrics before reading measurements. Outputs are
created in a separate, previously nonexistent directory. The source run is unchanged.
Each output records the input manifest digest and the generating script's digest.

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
