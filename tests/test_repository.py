import importlib.util
from pathlib import Path
import subprocess

import pytest


spec = importlib.util.spec_from_file_location(
    "check_repository", Path(__file__).parents[1] / "scripts/check_repository.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def test_local_records_and_machine_metadata_are_rejected():
    files = {
        "AGENTS.md": b"instructions",
        ".env": b"configuration",
        ".gstack/browser.json": b"{}",
        "docs/example.md": ("/" + "Users" + "/example/run").encode(),
        "results/test.xml": ('hostname="research-machine.' + 'local"').encode(),
    }
    findings = checks.inspect_files(files)
    assert len(findings) == 5
    assert all(any(f.startswith(name + ":") for f in findings) for name in files)


def test_scientific_configuration_and_binary_measurements_are_allowed():
    assert checks.inspect_files({
        "docs/optics.md": b"Commands are optical path difference in nanometres.",
        "results/data.npz": b"\x80\x81\x82",
        "configs/experiment.json": b'{"photons": 20000}',
        ".env.example": b"EXAMPLE_OPTION=",
    }) == []


def test_coordination_prose_is_rejected_in_documents():
    assert len(checks.inspect_files({"README.md": b"A research factory with hourly continuation."})) == 1


def test_check_reads_the_index_and_commit_not_unstaged_working_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    path = tmp_path / "README.md"
    path.write_text("Scientific model documentation.")
    subprocess.run(["git", "add", "README.md"], check=True)
    path.write_text("A research factory with hourly continuation.")
    assert checks.inspect_files(checks.tracked_files()) == []
    subprocess.run(["git", "add", "README.md"], check=True)
    assert checks.inspect_files(checks.tracked_files())
    subprocess.run(["git", "-c", "user.name=Test Maintainer", "-c", "user.email=test@example.invalid",
        "-c", "core.hooksPath=/dev/null", "commit", "-qm", "test: stage repository fixture"], check=True)
    assert checks.inspect_files(checks.tracked_files("HEAD"))


def test_tracked_symlinks_require_review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    (tmp_path / "link").symlink_to("external-file")
    subprocess.run(["git", "add", "link"], check=True)
    with pytest.raises(ValueError, match="non-regular tracked file"):
        checks.tracked_files()
