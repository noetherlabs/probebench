"""Independently compare retained camera/command prefixes across exposure budgets."""
import argparse
import json
from pathlib import Path

import numpy as np

from probebench.run import digest, write_json
from probebench.sequential import verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    manifest=verify_run(args.run)
    groups={}
    for path in sorted((args.run/"cases").glob("episode_*.json")):
        record=json.loads(path.read_text())
        case=record["case"]
        key=(case["method"],case["truth_index"],case["replicate"],case["photons_per_exposure"])
        groups.setdefault(key,[]).append(record)
    comparisons=[]
    for key,records in groups.items():
        records.sort(key=lambda r:r["case"]["exposure_budget"])
        reference=records[-1]
        for smaller in records[:-1]:
            count=smaller["case"]["exposure_budget"]
            for index in range(count):
                name=f"frame_{index:04d}.npz"
                left_path=args.run/"frames"/smaller["case"]["id"]/name
                right_path=args.run/"frames"/reference["case"]["id"]/name
                with np.load(left_path,allow_pickle=False) as left,np.load(right_path,allow_pickle=False) as right:
                    metadata_equal=(json.loads(left["record_json"].item())["observation"] ==
                                    json.loads(right["record_json"].item())["observation"])
                    images_equal=np.array_equal(left["image_electrons"],right["image_electrons"])
                if not metadata_equal or not images_equal:
                    raise ValueError(f"Changed paired measurement prefix: {key}, exposure {index}")
            comparisons.append({"episode_id":smaller["case"]["id"],
                "reference_episode_id":reference["case"]["id"],"identical_exposure_prefix":count})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,{"input_manifest_sha256":digest(args.run/"manifest.json"),
        "check_source_sha256":digest(Path(__file__)),"episode_count":manifest["episode_count"],
        "policy_fixture_noise_groups":len(groups),"comparisons":comparisons,
        "identical_prefix_exposures":sum(c["identical_exposure_prefix"] for c in comparisons),
        "scope":"Exact equality of raw images and public metadata across nested budgets; elapsed timings and fit logs are not expected to match."})


if __name__ == "__main__":
    main()
