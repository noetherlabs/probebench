import fcntl
import json
from types import SimpleNamespace

import pytest

from probebench import study as storage
from probebench.run import digest


@pytest.fixture
def admission(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    source = checkout / "src" / "probebench" / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n")
    relative = str(source.relative_to(checkout))
    inventory = {relative: digest(source)}
    monkeypatch.setattr(storage, "source_inventory", lambda: (checkout, inventory))
    config = {"operating_reserve_bytes": 4096}
    raw_bytes = 65536
    floor = config["operating_reserve_bytes"] + 16*1024**2
    required = floor + 3*raw_bytes
    capacity = SimpleNamespace(free=required)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: capacity)
    return SimpleNamespace(root=tmp_path / "run", config=config, raw_bytes=raw_bytes,
                           floor=floor, required=required, capacity=capacity,
                           source_relative=relative)


def start_partial(case):
    with storage.study_directory(case.root, case.config, case.raw_bytes):
        (case.root / "retained.bin").write_bytes(b"x" * case.raw_bytes)


def allocated_bytes(root):
    return sum(path.stat().st_blocks * 512 for path in root.rglob("*") if path.is_file())


def test_resume_credits_only_retained_bytes_under_lock_after_source_checks(admission):
    case = admission
    start_partial(case)
    credit = allocated_bytes(case.root)
    assert 0 < credit < 3*case.raw_bytes
    case.capacity.free = case.required - credit
    calls = []

    def validated_credit(root, study):
        calls.append(root)
        assert study == json.loads((root / "study.json").read_text())
        assert digest(root / "source" / case.source_relative) == study["source_sha256"][case.source_relative]
        with (root / ".lock").open("a") as second_lock:
            with pytest.raises(BlockingIOError):
                fcntl.flock(second_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert (root / "retained.bin").read_bytes() == b"x" * case.raw_bytes
        return allocated_bytes(root)

    with storage.study_directory(case.root, case.config, case.raw_bytes, resume=True,
                                 resume_credit=validated_credit):
        pass
    assert calls == [case.root]
    case.capacity.free -= 1
    with pytest.raises(OSError, match="Insufficient disk"):
        with storage.study_directory(case.root, case.config, case.raw_bytes, resume=True,
                                     resume_credit=validated_credit):
            pytest.fail("Insufficient remaining capacity must not yield")


def test_default_resume_still_requires_the_full_original_allowance(admission):
    case = admission
    start_partial(case)
    case.capacity.free = case.required - allocated_bytes(case.root)
    with pytest.raises(OSError, match="Insufficient disk"):
        with storage.study_directory(case.root, case.config, case.raw_bytes, resume=True):
            pytest.fail("Default admission must not silently credit files")
    case.capacity.free = case.required
    with storage.study_directory(case.root, case.config, case.raw_bytes, resume=True):
        pass


def test_excessive_credit_cannot_spend_reserve_or_fixed_headroom(admission):
    case = admission
    start_partial(case)
    case.capacity.free = case.floor - 1
    with pytest.raises(OSError, match="Insufficient disk"):
        with storage.study_directory(case.root, case.config, case.raw_bytes, resume=True,
                                     resume_credit=lambda root, study: 10*case.required):
            pytest.fail("Credit must preserve the reserve and headroom")
    case.capacity.free = case.floor
    with storage.study_directory(case.root, case.config, case.raw_bytes, resume=True,
                                 resume_credit=lambda root, study: 10*case.required):
        pass


@pytest.mark.parametrize("credit", [-1, True, False, 1.5, "4096", None])
def test_invalid_credit_is_rejected_before_yield(admission, credit):
    case = admission
    start_partial(case)
    with pytest.raises(ValueError, match="nonnegative integer"):
        with storage.study_directory(case.root, case.config, case.raw_bytes, resume=True,
                                     resume_credit=lambda root, study: credit):
            pytest.fail("Invalid credit must not yield")


@pytest.mark.parametrize("resume", [False, True])
def test_new_run_cannot_invoke_credit_or_bypass_full_admission(admission, resume):
    case = admission
    calls = []

    def forbidden(root, study):
        calls.append(root)
        raise AssertionError("A new run cannot claim retained-byte credit")

    case.capacity.free = case.required - 1
    with pytest.raises(OSError, match="Insufficient disk"):
        with storage.study_directory(case.root, case.config, case.raw_bytes, resume=resume,
                                     resume_credit=forbidden):
            pytest.fail("Insufficient new-run capacity must not yield")
    assert not case.root.exists() and calls == []
    case.capacity.free = case.required
    with storage.study_directory(case.root, case.config, case.raw_bytes, resume=resume,
                                 resume_credit=forbidden):
        pass
    assert calls == []


def test_existing_empty_directory_has_no_retained_binding_to_credit(admission):
    case = admission
    case.root.mkdir()
    case.capacity.free = case.required - 1

    def forbidden(root, study):
        raise AssertionError("An empty directory is not a bound study")

    with pytest.raises(OSError, match="Insufficient disk"):
        with storage.study_directory(case.root, case.config, case.raw_bytes, resume=True,
                                     resume_credit=forbidden):
            pytest.fail("Full admission is required for an unbound directory")


@pytest.mark.parametrize("damage", ["binding", "source"])
def test_binding_and_source_validation_precede_credit(admission, damage):
    case = admission
    start_partial(case)
    if damage == "binding":
        header = case.root / "study.json"
        saved = json.loads(header.read_text())
        saved["configuration"]["operating_reserve_bytes"] += 1
        header.write_text(json.dumps(saved))
    else:
        (case.root / "source" / case.source_relative).write_text("VALUE = 2\n")

    def forbidden(root, study):
        raise AssertionError("Invalid study must not reach credit callback")

    with pytest.raises(ValueError, match="changed"):
        with storage.study_directory(case.root, case.config, case.raw_bytes, resume=True,
                                     resume_credit=forbidden):
            pytest.fail("Invalid study must not yield")


def test_callback_failure_propagates_and_releases_lock(admission):
    case = admission
    start_partial(case)

    def failed(root, study):
        raise ValueError("Retained evidence failed validation")

    with pytest.raises(ValueError, match="Retained evidence"):
        with storage.study_directory(case.root, case.config, case.raw_bytes, resume=True,
                                     resume_credit=failed):
            pytest.fail("A failed evidence audit must not yield")
    with storage.study_directory(case.root, case.config, case.raw_bytes, resume=True,
                                 resume_credit=lambda root, study: 0):
        pass


def test_completed_manifest_path_does_not_invoke_resume_credit(admission):
    case = admission
    start_partial(case)
    # Completed archive verification is the caller's responsibility; this tests
    # only that the storage context does not audit or credit a completed run.
    (case.root / "manifest.json").write_text("{}\n")
    case.capacity.free = 0

    def forbidden(root, study):
        raise AssertionError("Completed runs do not need resume admission credit")

    with storage.study_directory(case.root, case.config, case.raw_bytes, resume=True,
                                 resume_credit=forbidden):
        pass
