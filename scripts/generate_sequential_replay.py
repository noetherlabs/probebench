"""Build a network-independent replay from a verified sequential experiment."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image
import numpy as np

from probebench.run import digest, write_json
from probebench.sequential import verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = verify_run(args.run)
    study = json.loads((args.run/"study.json").read_text())
    if len(study["configuration"]["photons_per_exposure"]) != 1:
        raise ValueError("This replay requires one declared photon level per exposure")
    template = Path(__file__).parent/"templates/sequential-replay.html"
    episodes = []
    for path in sorted((args.run/"cases").glob("episode_*.json")):
        record = json.loads(path.read_text())
        case, result = record["case"], record["result"]
        frames = []
        for index, trajectory in enumerate(result["trajectory"]):
            with np.load(args.run/"frames"/case["id"]/f"frame_{index:04d}.npz", allow_pickle=False) as saved:
                image = saved["image_electrons"]
                display = np.log1p(np.maximum(image,0))/np.log1p(case["photons_per_exposure"])
            buffer = io.BytesIO()
            matplotlib.image.imsave(buffer, display, format="png", origin="lower", cmap="magma", vmin=0, vmax=1)
            frames.append({"png":"data:image/png;base64,"+base64.b64encode(buffer.getvalue()).decode(),
                           "measurement":trajectory})
        episodes.append({"case":case,"frames":frames,
            "final":result["final"],"final_command":result["final_applied_opd_nm"],
            "measured_indices":result["final_command_observation_indices"],
            "initial":result["initial"],"fit_count":len(result["fit_records"]),
            "failed_fits":sum(f["fit_status"]=="numerical_failure" for f in result["fit_records"]),
            "unconverged_fits":sum(f["fit_status"]=="finite" and not f["optimizer_success"] for f in result["fit_records"])})
    data = {"episodes":episodes,"configuration":study["configuration"],
            "manifest_sha256":digest(args.run/"manifest.json"),"exposure_count":manifest["exposure_count"]}
    encoded = json.dumps(data,separators=(",",":"),allow_nan=False).replace("<","\\u003c")
    html = template.read_text().replace("%%DATA%%",encoded)
    script = html.split('<script id="replay-script">',1)[1].split("</script>",1)[0]
    script_hash = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    html = html.replace("%%SCRIPT_HASH%%",script_hash)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"index.html").write_text(html)
    write_json(args.output/"provenance.json",{
        "input_run":args.run.name,"input_manifest_sha256":digest(args.run/"manifest.json"),
        "episode_count":manifest["episode_count"],"exposure_count":manifest["exposure_count"],
        "generator_sha256":digest(Path(__file__)),"template_sha256":digest(template),
        "html_sha256":digest(args.output/"index.html"),
        "display":"log1p(max(raw electrons,0))/log1p(expected photons), magma, fixed limits 0..1",
        "scope":"Recorded development exposures only; final evaluator scores do not synthesize a measured image.",
    })


if __name__ == "__main__":
    main()
