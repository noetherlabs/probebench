import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
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


@pytest.mark.parametrize("name", [
    "AGENT.md", "docs/agents.MD", "Claude.md", "GEMINI.md", "plan.md",
    "internal/Status.Md", "decisions.md", ".github/copilot-instructions.md",
    ".cursorrules", ".windsurfrules", ".Claude/settings.json",
    ".cursor/rules/model.mdc", ".windsurf/rules/private.md", "secret.KEY",
])
def test_private_instruction_variants_are_rejected(name):
    assert checks.inspect_files({name: b"private"}) == [
        f"{name}: excluded local or sensitive file"]


@pytest.mark.parametrize("suffix", [".txt", ".rst", ".markdown", ".mdx", ".ipynb", ".adoc"])
def test_coordination_checks_cover_other_document_formats(suffix):
    assert checks.inspect_files({"notes" + suffix: b"Hourly continuation of a research factory."})


@pytest.mark.parametrize("path", [
    "C:" + "\\" + "Users" + "\\" + "researcher" + "\\" + "private",
    "C:" + "\\\\" + "Users" + "\\\\" + "researcher",
    "d:" + "/Users" + "/researcher",
])
def test_windows_home_paths_are_rejected(path):
    assert checks.inspect_files({"trace.json": path.encode()}) == [
        "trace.json: personal filesystem path or local hostname"]


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


def test_hostname_detection_preserves_boundaries_without_long_prefix_retries():
    previous = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.local\b")
    rng = random.Random(164)
    alphabet = "aZ09_.-/ \néα"
    cases = ["", ".", "---", "a", "0.-", "é", "a/", "a."]
    cases += ["".join(rng.choices(alphabet, k=rng.randrange(80))) for _ in range(2000)]
    for prefix in cases:
        for ending in ("", " ", "/", ".", "_", "a", "é"):
            value = prefix + ".local" + ending
            assert bool(checks.LOCAL_HOST.search(value)) == bool(previous.search(value))
    assert checks.LOCAL_HOST.search("a" * 1_000_000) is None


def test_batch_preserves_binary_bytes_and_unusual_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    payload = b"\x00\xff\n012345 blob 17\n" * 8192
    expected = {
        "measurement one.bin": payload,
        "measurement\ttwo\n.bin": payload,
        "empty.bin": b"",
        "μ.txt": "Optical phase\n".encode(),
    }
    for name, content in expected.items():
        (tmp_path / name).write_bytes(content)
    subprocess.run(["git", "add", "--all"], check=True)
    original_popen = subprocess.Popen
    commands = []

    def record_process(command, *args, **kwargs):
        commands.append(command)
        return original_popen(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", record_process)
    actual = checks.tracked_files()
    assert actual == expected
    assert actual["measurement one.bin"] is actual["measurement\ttwo\n.bin"]
    assert [command for command in commands if command[:2] == ["git", "cat-file"]] == [
        ["git", "cat-file", "--batch"]]


def test_empty_index_has_no_blob_process(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    assert checks.tracked_files() == {}
    assert checks.read_blobs([]) == {}


def test_missing_blob_fails_closed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    with pytest.raises(ValueError, match="invalid Git blob response"):
        checks.read_blobs(["0" * 40])


def test_tree_object_is_rejected_as_blob(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    tree = subprocess.check_output(["git", "write-tree"], text=True).strip()
    with pytest.raises(ValueError, match="invalid Git blob response"):
        checks.read_blobs([tree])


def test_unmerged_index_fails_before_blob_scan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init", "-q"], check=True)
    blob = subprocess.check_output(["git", "hash-object", "-w", "--stdin"], input=b"phase\n").decode().strip()
    subprocess.run(["git", "update-index", "--index-info"],
        input=f"100644 {blob} 1\tconflict.txt\n".encode(), check=True)
    with pytest.raises(ValueError, match="unresolved index entry"):
        checks.tracked_files()


def hook_repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "scripts").mkdir()
    shutil.copyfile(Path(__file__).parents[1] / "scripts/check_repository.py",
                    root / "scripts/check_repository.py")
    binary = tmp_path / "bin"
    binary.mkdir()
    scanner = binary / "gitleaks"
    scanner.write_text(
        "#!/usr/bin/env python3\nimport json,os,sys\n"
        "with open(os.environ['SCAN_LOG'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "sys.exit(int(os.environ.get('SCAN_EXIT', '0')))\n")
    scanner.chmod(0o755)
    log = tmp_path / "scan.jsonl"
    env = {**os.environ, "PATH": str(binary) + os.pathsep + os.environ["PATH"],
           "SCAN_LOG": str(log)}
    (root / "README.md").write_text("Scientific methods.\n")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    base = fixture_commit(root)
    return root, base, env, log


def fixture_commit(root):
    subprocess.run(["git", "-c", "user.name=Test Maintainer",
        "-c", "user.email=test@example.invalid", "-c", "core.hooksPath=/dev/null",
        "commit", "-qm", "test: retain fixture"], cwd=root, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def invoke_hook(root, name, env, local=None, remote=None):
    record = "" if local is None else f"refs/heads/main {local} refs/heads/main {remote}\n"
    return subprocess.run(["sh", str(Path(__file__).parents[1] / ".githooks" / name)],
        input=record, cwd=root, env=env, capture_output=True, text=True)


def test_commit_hook_checks_staged_snapshot_and_credential_scan(tmp_path):
    root, _, env, log = hook_repository(tmp_path)
    readme = root / "README.md"
    readme.write_text("A research factory with hourly continuation.\n")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    readme.write_text("Scientific methods.\n")
    result = invoke_hook(root, "pre-commit", env)
    assert result.returncode != 0
    assert "internal coordination" in result.stderr
    assert not log.exists()
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    readme.write_text("A research factory with hourly continuation.\n")
    result = invoke_hook(root, "pre-commit", env)
    assert result.returncode == 0, result.stderr
    assert json.loads(log.read_text()) == [
        "git", ".", "--pre-commit", "--staged", "--redact", "--no-banner"]
    assert invoke_hook(root, "pre-commit", {**env, "SCAN_EXIT": "1"}).returncode != 0


@pytest.mark.parametrize("new_ref", [False, True])
def test_push_rejects_removed_private_files_even_if_another_remote_has_commits(tmp_path, new_ref):
    root, base, env, log = hook_repository(tmp_path)
    (root / "AGENT.md").write_text("private instructions\n")
    subprocess.run(["git", "add", "AGENT.md"], cwd=root, check=True)
    fixture_commit(root)
    subprocess.run(["git", "rm", "-q", "AGENT.md"], cwd=root, check=True)
    tip = fixture_commit(root)
    subprocess.run(["git", "update-ref", "refs/remotes/another/main", tip], cwd=root, check=True)
    result = invoke_hook(root, "pre-push", env, tip, "0" * 40 if new_ref else base)
    assert result.returncode != 0
    assert "AGENT.md: excluded" in result.stderr
    assert not log.exists()


@pytest.mark.parametrize("new_ref", [False, True])
def test_push_scans_exact_destination_range_and_propagates_scanner_failure(tmp_path, new_ref):
    root, base, env, log = hook_repository(tmp_path)
    (root / "README.md").write_text("Updated scientific methods.\n")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    tip = fixture_commit(root)
    remote = "0" * 40 if new_ref else base
    result = invoke_hook(root, "pre-push", env, tip, remote)
    assert result.returncode == 0, result.stderr
    revision = tip if new_ref else f"{base}..{tip}"
    assert json.loads(log.read_text()) == [
        "git", ".", "--redact", "--no-banner", "--log-opts=" + revision]
    assert invoke_hook(root, "pre-push", {**env, "SCAN_EXIT": "1"}, tip, remote).returncode != 0


def test_push_refuses_unknown_destination_and_skips_deletion(tmp_path):
    root, base, env, log = hook_repository(tmp_path)
    result = invoke_hook(root, "pre-push", env, base, "1" * 40)
    assert result.returncode != 0
    assert not log.exists()
    result = invoke_hook(root, "pre-push", env, "0" * 40, base)
    assert result.returncode == 0, result.stderr
    assert not log.exists()
