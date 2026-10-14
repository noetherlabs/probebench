"""Export verified development measurements to local ImageFolder and raw-array files."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image
import numpy as np

from probebench.run import digest, verify_files, write_json
from probebench.sequential import verify_run


def export(run, output):
    run, output = Path(run), Path(output)
    manifest = verify_run(run)
    study = json.loads((run/"study.json").read_text())
    output.mkdir(parents=True,exist_ok=False)
    images, raw = output/"images", output/"raw"
    images.mkdir(); raw.mkdir()
    rows, episodes = [], []
    for path in sorted((run/"cases").glob("episode_*.json")):
        record = json.loads(path.read_text())
        case, result = record["case"], record["result"]
        episodes.append({"episode_id":case["id"],"case":case,"result":result})
        for index, measured in enumerate(result["trajectory"]):
            source = run/"frames"/case["id"]/f"frame_{index:04d}.npz"
            identity = f'{case["id"]}_frame_{index:04d}'
            with np.load(source,allow_pickle=False) as saved:
                electrons = saved["image_electrons"].copy()
            display = np.log1p(np.maximum(electrons,0))/np.log1p(case["photons_per_exposure"])
            matplotlib.image.imsave(images/f"{identity}.png",display,origin="lower",cmap="magma",vmin=0,vmax=1)
            # Raw arrays exclude estimator/evaluator state and retain signed detector values.
            np.save(raw/f"{identity}.npy",electrons,allow_pickle=False)
            rows.append({"file_name":f"{identity}.png","episode_id":case["id"],
                "cohort":"public_development","method":case["method"],
                "truth_index":case["truth_index"],"replicate":case["replicate"],
                "exposure_index":index,"exposure_budget":case["exposure_budget"],
                "expected_photons":measured["expected_photons"],
                "cumulative_expected_photons":measured["cumulative_expected_photons"],
                "applied_opd_nm":measured["applied_opd_nm"],
                "evaluator_wfe_nm":measured["residual_wfe_nm"],
                "evaluator_strehl_peak_ratio":measured["strehl_peak_ratio"],
                "raw_array":f"raw/{identity}.npy","raw_array_sha256":digest(raw/f"{identity}.npy"),
                "source_frame_sha256":digest(source)})
    for path, entries in ((images/"metadata.jsonl",rows),(output/"episodes.jsonl",episodes)):
        path.write_text("".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n" for r in entries))
    write_json(output/"configuration.json",study["configuration"])
    card = f'''---
language:
- en
pretty_name: ProbeBench development measurements
tags:
- optics
- wavefront-sensing
- scientific-computing
configs:
- config_name: default
  data_files:
  - split: development
    path: images/**
---

# ProbeBench development measurements

This export contains {manifest["episode_count"]} synthetic calibration episodes and
{manifest["exposure_count"]} acquired focal-plane images. The wavefronts are known
development fixtures; there is no held-out evaluation split or multimodal-agent result.

## Contents

`images/metadata.jsonl` links each fixed-scale display PNG to its applied OPD command,
photon accounting, method and episode identity. Columns prefixed `evaluator_` are latent
simulation-derived labels, not controller inputs. `raw/` contains the corresponding
signed float64 electron arrays in NumPy format. Raw paths are relative to the export
root. `episodes.jsonl` contains full evaluator trajectories, final submission status,
latent wavefronts, detector seeds and optimizer diagnostics. Do not pass that file or
evaluator metadata to a controller during evaluation.

The PNGs use log1p(max(raw electrons, 0)) / log1p(expected photons), a fixed 0–1 range and
the magma colour map. They are display views; inference should use the signed arrays.
No image is synthesized for a final command lacking a recorded exposure.

## Load locally

```python
from datasets import load_dataset
from pathlib import Path
import numpy as np

root = Path("work/imagefolder")
frames = load_dataset("imagefolder", data_files={{"development": str(root / "images" / "**")}},
                      split="development")
electrons = np.load(root / frames[0]["raw_array"], allow_pickle=False)
```

The export follows the documented [ImageFolder metadata format](https://huggingface.co/docs/datasets/image_dataset).

## Model and limitations

The plant is static, monochromatic scalar Fourier optics with a circular pupil, six
sampled Zernike modes and Poisson–Gaussian detector noise. Expected incident photons
are charged per exposure. Final corrections can be submitted without an additional
measurement; evaluator scores at an unmeasured command are not camera evidence.
The estimator assumes exactly matched optical calibration. Shared noise streams and
related wavefronts make budget points correlated. This small public cohort supports
implementation analysis, not population rankings or telescope-performance claims.

The configuration, complete episode outcomes and manifest preserve provenance and
non-winning outcomes. Hashes establish file consistency, not author authentication.
The source experiment manifest is `{digest(run/"manifest.json")}`.
'''
    (output/"README.md").write_text(card)
    write_json(output/"manifest.json",{
        "schema":1,"purpose":"imagefolder_development_export",
        "input_manifest_sha256":digest(run/"manifest.json"),
        "generator_sha256":digest(Path(__file__)),
        "episode_count":manifest["episode_count"],"exposure_count":manifest["exposure_count"],
        "files":{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
    })
    verify_files(output)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    export(args.run,args.output)
