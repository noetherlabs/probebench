"""Content-integrity boundary tests; fixture directories are not real installations."""
import base64
import csv
import hashlib
import json
import shutil

import pytest

from probebench import provenance


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_files(root):
    record = root / "noether_probebench-0.0.1.dist-info/RECORD"
    with record.open("w", newline="") as stream:
        writer = csv.writer(stream)
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != record:
                value = base64.urlsafe_b64encode(bytes.fromhex(sha(path))).rstrip(b"=").decode()
                writer.writerow([path.relative_to(root).as_posix(), "sha256=" + value, path.stat().st_size])
        writer.writerow([record.relative_to(root).as_posix(), "", ""])


@pytest.fixture
def installed(tmp_path):
    root = tmp_path / "site-packages"
    package = root / "probebench"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Fixture package."""\n')
    (package / "run.py").write_text("VALUE = 1\n")
    info = root / "noether_probebench-0.0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.3\nName: noether-probebench\nVersion: 0.0.1\n")
    (info / "WHEEL").write_text("Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
    (info / "INSTALLER").write_text("pip\n")
    record_files(root)
    return root, package, info


def test_source_layout_keeps_historical_relative_paths(tmp_path):
    package = tmp_path / "src/probebench"
    package.mkdir(parents=True)
    for name in ("__init__.py", "run.py"):
        (package / name).write_text("# historical source fixture\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='noether-probebench'\n")
    (tmp_path / "uv.lock").write_text("version=1\n")
    root, inventory = provenance.source_inventory(package)
    assert root == tmp_path
    expected = ["src/probebench/__init__.py", "src/probebench/run.py", "pyproject.toml", "uv.lock"]
    assert inventory == {name: sha(tmp_path / name) for name in expected}
    assert provenance.identity_fields(inventory) == {}
    provenance.verify_identity({"source_sha256": inventory})


def test_installed_inventory_records_actual_files_and_explicit_scheme(installed):
    root, package, info = installed
    collected_root, inventory = provenance.source_inventory(package)
    assert collected_root == root
    assert inventory == {p.relative_to(root).as_posix(): sha(p) for p in root.rglob("*") if p.is_file()}
    assert "pyproject.toml" not in inventory
    assert not any(name.startswith("src/") for name in inventory)
    fields = provenance.identity_fields(inventory)
    assert fields == {"provenance": {"scheme": provenance.INSTALLED_SCHEME,
                      "distribution": "noether-probebench", "metadata_directory": info.name,
                      "omitted_metadata_content": [info.name + "/direct_url.json"]}}
    provenance.verify_identity({"source_sha256": inventory, **fields})


def test_record_cannot_replace_its_own_metadata_with_another_distribution(installed):
    root, package, info = installed
    foreign = root / "another_distribution-0.0.1.dist-info"
    foreign.mkdir()
    (foreign / "METADATA").write_text("Metadata-Version: 2.3\nName: another-distribution\nVersion: 0.0.1\n")
    (foreign / "WHEEL").write_bytes((info / "WHEEL").read_bytes())
    rows = []
    for path in [*package.glob("*.py"), foreign / "METADATA", foreign / "WHEEL"]:
        encoded = base64.urlsafe_b64encode(bytes.fromhex(sha(path))).rstrip(b"=").decode()
        rows.append([path.relative_to(root).as_posix(), "sha256=" + encoded, path.stat().st_size])
    rows.append([foreign.name + "/RECORD", "", ""])
    with (foreign / "RECORD").open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    # The selected distribution's RECORD must not nominate this self-consistent
    # foreign metadata tree while dropping the METADATA that selected it.
    (info / "RECORD").write_bytes((foreign / "RECORD").read_bytes())
    with pytest.raises(ValueError, match="unrelated installed RECORD path"):
        provenance.source_inventory(package)


@pytest.mark.parametrize("damage", ["module", "metadata", "missing_record", "missing_module",
                                    "extra_module", "extra_metadata", "duplicate", "escape",
                                    "wrong_size", "record_self_hash", "symlink", "editable", "ambiguous",
                                    "empty_path", "malformed_origin"])
def test_inconsistent_or_ambiguous_installations_are_refused(installed, damage):
    root, package, info = installed
    record = info / "RECORD"
    if damage == "module":
        (package / "run.py").write_text("VALUE = 2\n")
    elif damage == "metadata":
        (info / "WHEEL").write_text("Changed wheel metadata\n")
    elif damage == "missing_record":
        record.unlink()
    elif damage == "missing_module":
        (package / "run.py").unlink()
    elif damage == "extra_module":
        (package / "unrecorded.py").write_text("VALUE = 3\n")
    elif damage == "extra_metadata":
        (info / "direct_url.json").write_text('{"dir_info":{"editable":true}}')
    elif damage == "duplicate":
        with record.open("a") as stream:
            stream.write(record.read_text().splitlines()[0] + "\n")
    elif damage == "escape":
        with record.open("a") as stream:
            stream.write("../outside.py,,\n")
    elif damage == "wrong_size":
        rows = list(csv.reader(record.read_text().splitlines()))
        rows[0][2] = "999"
        with record.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows)
    elif damage == "record_self_hash":
        record.write_text(record.read_text().replace("RECORD,,", "RECORD,sha256=invalid,1"))
    elif damage == "symlink":
        (package / "run.py").unlink()
        (root.parent / "outside.py").write_text("VALUE = 1\n")
        (package / "run.py").symlink_to(root.parent / "outside.py")
    elif damage == "editable":
        (info / "direct_url.json").write_text(json.dumps({"dir_info": {"editable": True}}))
        record_files(root)
    elif damage == "ambiguous":
        shutil.copytree(info, root / "noether_probebench-0.0.2.dist-info")
    elif damage == "empty_path":
        with record.open("a") as stream:
            stream.write(".,,\n")
    elif damage == "malformed_origin":
        (info / "direct_url.json").write_text("[]")
        record_files(root)
    with pytest.raises(ValueError):
        provenance.source_inventory(package)


def test_interpreter_cache_is_not_source_but_record_bytes_are_retained(installed):
    root, package, info = installed
    cache = package / "__pycache__/run.cpython-311.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"inert cache fixture")
    with (info / "RECORD").open("a") as stream:
        stream.write("probebench/__pycache__/run.cpython-311.pyc,,\n")
    _, inventory = provenance.source_inventory(package)
    assert cache.relative_to(root).as_posix() not in inventory
    assert inventory[info.name + "/RECORD"] == sha(info / "RECORD")


@pytest.mark.parametrize("origin", ["file:///private/fixture/research",
                                   "https://fixture-user:fixture-password@example.invalid/private.git"])
def test_private_installation_origin_is_checked_but_never_copied(installed, tmp_path, origin):
    root, package, info = installed
    location = info / "direct_url.json"
    location.write_text(json.dumps({"url": origin}))
    record_files(root)
    record_bytes = (info / "RECORD").read_bytes()
    collected_root, inventory = provenance.source_inventory(package)
    assert location.relative_to(root).as_posix() not in inventory
    snapshot = tmp_path / "retained-source"
    for relative in inventory:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(collected_root / relative, destination)
    assert (snapshot / info.name / "RECORD").read_bytes() == record_bytes
    assert "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(sha(location))).rstrip(b"=").decode() in record_bytes.decode()
    assert not (snapshot / info.name / "direct_url.json").exists()
    assert all(origin.encode() not in path.read_bytes() for path in snapshot.rglob("*") if path.is_file())
    location.write_text(json.dumps({"url": origin + "/changed"}))
    with pytest.raises(ValueError, match="differs from wheel RECORD"):
        provenance.source_inventory(package)


@pytest.mark.parametrize("damage", ["omitted", "scheme", "mixed", "incomplete"])
def test_installed_scheme_cannot_be_omitted_or_changed(installed, damage):
    _, package, _ = installed
    _, inventory = provenance.source_inventory(package)
    value = {"source_sha256": inventory, **provenance.identity_fields(inventory)}
    if damage == "omitted":
        del value["provenance"]
    elif damage == "scheme":
        value["provenance"]["scheme"] = "source-checkout"
    elif damage == "mixed":
        inventory["src/probebench/run.py"] = inventory["probebench/run.py"]
    else:
        del inventory["probebench/run.py"]
    with pytest.raises(ValueError):
        provenance.verify_identity(value)


@pytest.mark.parametrize("path", ["../outside", "/outside", "src/../../outside", "src//run.py", ".", ""])
def test_retained_source_paths_are_validated_before_access(tmp_path, monkeypatch, path):
    from probebench import qualification, run
    inventory = {path: "0" * 64}
    (tmp_path / "study.json").write_text(json.dumps({"source_sha256": inventory}))
    manifest = {"schema": 1, "source_sha256": inventory, "files": {}}
    monkeypatch.setattr(qualification, "verify_files", lambda root: {
        **manifest, "purpose": "estimator_development_qualification"})
    monkeypatch.setattr(run, "verify_files", lambda root: {**manifest, "purpose": "development_only"})

    def forbidden(*args):
        raise AssertionError("Verifier tried to read an unvalidated source path")

    monkeypatch.setattr(qualification, "digest", forbidden)
    with pytest.raises(ValueError, match="provenance file path"):
        qualification.verify_qualification(tmp_path)
    with pytest.raises(ValueError, match="provenance file path"):
        run.verify_run(tmp_path)


def test_source_identity_must_bind_an_actual_verified_snapshot(tmp_path, monkeypatch):
    from probebench import qualification, run
    inventory = {"src/probebench/run.py": "0" * 64}
    (tmp_path / "study.json").write_text(json.dumps({"source_sha256": inventory}))
    (tmp_path / "summary.json").write_text("[]")
    manifest = {"schema": 1, "source_sha256": inventory, "files": {}, "episodes": 0}
    monkeypatch.setattr(qualification, "verify_files", lambda root: {
        **manifest, "purpose": "estimator_development_qualification"})
    monkeypatch.setattr(run, "verify_files", lambda root: {**manifest, "purpose": "development_only"})
    with pytest.raises(ValueError, match="Source snapshot disagrees"):
        qualification.verify_qualification(tmp_path)
    with pytest.raises(ValueError, match="Source snapshot disagrees"):
        run.verify_run(tmp_path)
