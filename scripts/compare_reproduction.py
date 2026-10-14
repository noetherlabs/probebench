"""Verify an exact development reproduction; the creation timestamp may differ."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from probebench.run import digest, verify_run, write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("original", type=Path)
parser.add_argument("reproduction", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
original = verify_run(args.original)
reproduction = verify_run(args.reproduction)
for key in set(original) | set(reproduction):
    if key != "created_utc" and original.get(key) != reproduction.get(key):
        raise ValueError(f"Reproduction differs in {key}")
write_json(args.output, {
    "verified_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "exact retained development reproduction in a newly installed local environment",
    "original": str(args.original), "reproduction": str(args.reproduction),
    "original_manifest_sha256": digest(args.original / "manifest.json"),
    "reproduction_manifest_sha256": digest(args.reproduction / "manifest.json"),
    "runtime": reproduction["runtime"], "episodes": original["episodes"],
    "identical_file_count": len(original["files"]),
    "identical_file_sha256": original["files"],
    "only_permitted_manifest_difference": "created_utc", "passed": True,
})
print(f"Verified {original['episodes']} episodes and {len(original['files'])} identical files.")
