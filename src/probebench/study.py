"""Local study admission, exclusive ownership and immutable source checkpoints."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil

import numpy as np
import scipy

from .qualification import publish_json
from .run import digest, source_inventory, verify_files


@contextmanager
def study_directory(root, configuration, raw_bytes, resume=False):
    root = Path(root)
    source_root, sources = source_inventory()
    study = {"configuration": configuration, "source_sha256": sources,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "platform": platform.platform(),
                    "machine": platform.machine()}}
    reserve = configuration["operating_reserve_bytes"]
    if type(reserve) is not int or reserve < 0:
        raise ValueError("Operating reserve must be a nonnegative integer")
    required = reserve + 3*raw_bytes + 16*1024**2
    root.parent.mkdir(parents=True, exist_ok=True)
    if not root.exists() and shutil.disk_usage(root.parent).free < required:
        raise OSError("Insufficient disk for retained study plus operating reserve")
    if root.is_symlink():
        raise ValueError("Study root must not be a symbolic link")
    root.mkdir(exist_ok=resume)
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Study directories must not contain symbolic links")
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        header = root / "study.json"
        if header.exists():
            if json.loads(header.read_text()) != study:
                raise ValueError("Source, configuration or runtime changed; resume refused")
        else:
            if any(p.name not in {".lock", "study.json.pending"} for p in root.iterdir()):
                raise ValueError("Cannot adopt a directory without a study binding")
            publish_json(header, study)
        for relative, expected in sources.items():
            dest = root / "source" / relative
            if dest.is_symlink() or not dest.resolve().is_relative_to(root.resolve()):
                raise ValueError("Source snapshot path escapes study directory")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if digest(dest) != expected:
                    raise ValueError("Source snapshot changed; resume refused")
            else:
                temporary = dest.with_suffix(dest.suffix + ".pending")
                shutil.copyfile(source_root / relative, temporary)
                os.replace(temporary, dest)
        if not (root / "manifest.json").exists() and shutil.disk_usage(root.parent).free < required:
            raise OSError("Insufficient disk for continued study and operating reserve")
        yield study


def publish_manifest(root, study, purpose, **counts):
    if source_inventory()[1] != study["source_sha256"]:
        raise RuntimeError("Source changed during acquisition; final publication refused")
    manifest = {"schema": 1, "purpose": purpose, "source_sha256": study["source_sha256"],
        **counts, "files": {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
                           if p.is_file() and p.name != "manifest.json.pending"}}
    if any(name.endswith(".pending") for name in manifest["files"]):
        raise ValueError("Unfinished publication buffers remain")
    publish_json(root / "manifest.json", manifest)
    return manifest


def verify_study(root, purpose):
    root = Path(root)
    manifest = verify_files(root)
    if manifest.get("schema") != 1 or manifest.get("purpose") != purpose:
        raise ValueError("Unexpected study purpose or schema")
    study = json.loads((root / "study.json").read_text())
    if manifest["source_sha256"] != study["source_sha256"]:
        raise ValueError("Manifest/source binding mismatch")
    for relative, expected in study["source_sha256"].items():
        name = PurePosixPath(relative)
        artifact = f"source/{relative}"
        if (name.is_absolute() or ".." in name.parts or str(name) != relative
            or artifact not in manifest["files"]):
            raise ValueError("Unsafe source snapshot path")
        if manifest["files"][artifact] != expected:
            raise ValueError("Source snapshot mismatch")
    return manifest, study
