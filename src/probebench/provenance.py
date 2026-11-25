"""Content identities for source layouts and ordinary wheel installations.

Installed identities describe project files and selected distribution metadata
on this host. Origin metadata is checked locally but not copied; the unchanged
RECORD retains its digest. RECORD consistency is not publisher authentication,
a clean Git checkout, or an identity for the original wheel archive.
"""
from __future__ import annotations

import base64
import csv
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path, PurePosixPath
import re


INSTALLED_SCHEME = "probebench.installed-wheel-record/1"
DISTRIBUTION = "noether-probebench"


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def _bytecode(name):
    return "__pycache__" in name.parts and name.suffix == ".pyc"


def _regular(root, name):
    path = root.joinpath(*name.parts)
    if (not path.is_file() or not path.resolve().is_relative_to(root)
            or any(p.is_symlink() for p in (path, *path.parents) if p != root.parent)):
        raise ValueError("Installed provenance requires regular files inside the distribution root")
    return path


def _installed_inventory(package):
    root = package.parent
    candidates = []
    for directory in sorted(root.glob("*.dist-info")):
        if directory.is_symlink() or not directory.is_dir():
            continue
        distribution = metadata.Distribution.at(directory)
        if _name(distribution.metadata.get("Name", "")) == DISTRIBUTION:
            candidates.append((directory, distribution))
    if len(candidates) != 1:
        raise ValueError("Installed ProbeBench distribution is missing or ambiguous")
    directory, distribution = candidates[0]
    metadata_directory = directory.name
    record_name = PurePosixPath(metadata_directory) / "RECORD"
    if distribution.locate_file("probebench").resolve() != package:
        raise ValueError("Installed metadata does not own the executing package")
    record = _regular(root, record_name).read_text(encoding="utf-8")
    inventory, declared, package_files = {}, set(), set()
    for row in csv.reader(io.StringIO(record), strict=True):
        if len(row) != 3:
            raise ValueError("Invalid installed RECORD row")
        name, encoded_hash, size = row
        relative = PurePosixPath(name)
        if (not name or not relative.parts or str(relative) != name or relative.is_absolute()
                or ".." in relative.parts or "\\" in name
                or any(ord(c) < 32 for c in name)
                or relative.parts[0] not in {"probebench", metadata_directory}
                or name in declared):
            raise ValueError("Unsafe, duplicate or unrelated installed RECORD path")
        declared.add(name)
        if _bytecode(relative):
            continue  # Interpreter caches are not source or distribution metadata.
        path = _regular(root, relative)
        actual = _digest(path)
        if relative == record_name:
            if encoded_hash or size:
                raise ValueError("RECORD must leave its own hash and size empty")
        else:
            expected = "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(actual)).rstrip(b"=").decode()
            if encoded_hash != expected or size != str(path.stat().st_size):
                raise ValueError("Installed file differs from wheel RECORD")
        inventory[name] = actual
        if relative.parts[0] == "probebench":
            package_files.add(name)
    required = {"probebench/__init__.py", "probebench/run.py",
                *(metadata_directory + "/" + name for name in ("METADATA", "WHEEL", "RECORD"))}
    if not required <= inventory.keys():
        raise ValueError("Incomplete installed project or wheel metadata")
    actual_package_files = set()
    for path in package.rglob("*"):
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if path.is_symlink():
            raise ValueError("Installed package contains a symbolic link")
        if path.is_file() and not _bytecode(relative):
            actual_package_files.add(str(relative))
    if actual_package_files != package_files:
        raise ValueError("Installed package has missing or unrecorded files")
    actual_metadata_files = set()
    for path in (root / metadata_directory).rglob("*"):
        if path.is_symlink():
            raise ValueError("Installed metadata contains a symbolic link")
        if path.is_file():
            actual_metadata_files.add(path.relative_to(root).as_posix())
    if actual_metadata_files != {name for name in inventory if name.startswith(metadata_directory + "/")}:
        raise ValueError("Installed metadata has missing or unrecorded files")
    direct_url = metadata_directory + "/direct_url.json"
    if direct_url in inventory:
        origin = json.loads((root / direct_url).read_text(encoding="utf-8"))
        if type(origin) is not dict or type(origin.get("dir_info", {})) is not dict:
            raise ValueError("Invalid installed origin metadata")
        if origin.get("dir_info", {}).get("editable") is True:
            raise ValueError("Editable installations are not installed-wheel provenance")
        # Installation origins can contain private paths or authenticated URLs.
        # Keep the original RECORD digest entry, never substitute redacted bytes.
        del inventory[direct_url]
    return root, dict(sorted(inventory.items()))


def source_inventory(package):
    """Keep historical source paths; otherwise bind the actual installed files."""
    package = Path(package).resolve()
    root = package.parent.parent
    if package.parent.name == "src" and (root / "pyproject.toml").is_file():
        paths = sorted(package.glob("*.py")) + [root / "pyproject.toml"]
        if (root / "uv.lock").exists():
            paths.append(root / "uv.lock")
        return root, {str(path.relative_to(root)): _digest(path) for path in paths}
    return _installed_inventory(package)


def identity_fields(inventory):
    """Installed identities are explicit; historical source records stay unchanged."""
    if type(inventory) is not dict:
        raise ValueError("Invalid provenance file inventory")
    for name, digest in inventory.items():
        if type(name) is not str:
            raise ValueError("Invalid provenance file path")
        path = PurePosixPath(name)
        if (not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name
                or str(path) != name or any(ord(c) < 32 for c in name)):
            raise ValueError("Unsafe provenance file path")
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Invalid provenance file digest")
    directories = {PurePosixPath(name).parts[0] for name in inventory
                   if PurePosixPath(name).parts[0].endswith(".dist-info")}
    installed_paths = any(name.startswith("probebench/") for name in inventory)
    if not directories and not installed_paths:
        return {}
    if len(directories) != 1:
        raise ValueError("Ambiguous installed provenance identity")
    directory = next(iter(directories))
    required = {"probebench/__init__.py", "probebench/run.py",
                *(directory + "/" + name for name in ("METADATA", "WHEEL", "RECORD"))}
    if (not required <= inventory.keys()
            or any(PurePosixPath(name).parts[0] not in {"probebench", directory} for name in inventory)):
        raise ValueError("Incomplete or mixed installed provenance identity")
    return {"provenance": {"scheme": INSTALLED_SCHEME, "distribution": DISTRIBUTION,
                           "metadata_directory": directory,
                           "omitted_metadata_content": [directory + "/direct_url.json"]}}


def verify_identity(record):
    expected = identity_fields(record["source_sha256"])
    if (("provenance" in record) != bool(expected)
            or record.get("provenance") != expected.get("provenance")):
        raise ValueError("Missing or changed installed provenance scheme")
