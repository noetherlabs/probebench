"""Resume admission must trust replayed selection evidence, never file presence alone."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from probebench import selection, sequential, study as storage
from probebench.qualification import canonical_hash
from probebench.run import digest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trial(tmp_path, monkeypatch):
    config = json.loads((ROOT / "configs/classical_selection_v1.json").read_text())
    cohort = config["cohort"]
    # Reuse the public unit-test wavefront namespace; this is not a held-out cohort.
    cohort["seed_hex"] = "11" * 32
    cohort["specification"].update(
        cohort_id="selection_unit_fixture",
        optics={"pupil_pixels": 12, "oversampling": 1, "wavelength_nm": 550.0},
        strata=[{"id": "low", "rms_min_nm": 10.0, "rms_max_nm": 25.0, "base_count": 1}],
    )
    cohort["exclusions"]["optics"] = deepcopy(cohort["specification"]["optics"])
    cohort["exclusions"]["wavefronts"] = cohort["exclusions"]["wavefronts"][:3]
    config.update(
        excluded_cohorts=[], replicates=1, photons_per_exposure=[2000],
        exposure_budgets=[3], operating_reserve_bytes=4096,
        candidates=[{"id": "random_amp_020", "family": "random_probe_v2",
                     "parameters": {"random_amplitude_nm": 20.0}}],
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    # Exercise source binding and its retained bytes without depending on concurrent
    # checkout edits. Both acquisition and final publication inspect this inventory.
    checkout = tmp_path / "source_fixture"
    source = checkout / "src" / "probebench" / "fixture.py"
    source.parent.mkdir(parents=True)
    source.write_text('PURPOSE = "selection resume admission test"\n')
    inventory = {str(source.relative_to(checkout)): digest(source)}
    for module in (storage, sequential):
        monkeypatch.setattr(module, "source_inventory", lambda: (checkout, inventory))

    prepared = selection.prepare(config)
    estimate = selection.storage_estimate(config, prepared)
    assert estimate["episodes"] == 2 and estimate["exposures"] == 6
    capacity = SimpleNamespace(free=estimate["required_free_bytes"])
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path: capacity)
    return SimpleNamespace(
        root=tmp_path / "study", path=config_path, config=config, prepared=prepared,
        cases=selection.cases_for(config, prepared), capacity=capacity,
        required=estimate["required_free_bytes"],
        floor=config["operating_reserve_bytes"] + 16 * 1024**2,
    )


def fingerprints(root):
    return {str(path.relative_to(root)): digest(path)
            for path in root.rglob("*") if path.is_file()}


def retained_bytes(root):
    # These fixtures have no aliases or sparse files. Pending buffers and the
    # mutable lock file must not lower the free-space requirement. Call this on
    # the normal writer's output, before adding adversarial container padding.
    paths = [path for path in root.rglob("*")
             if path.is_file() and path.name != ".lock"
             and not path.name.endswith(".pending")]
    stats = [path.stat() for path in paths]
    assert len({(stat.st_dev, stat.st_ino) for stat in stats}) == len(stats)
    return sum(min(stat.st_size, stat.st_blocks * 512) for stat in stats)


def start_partial(trial):
    assert selection.run(trial.root, trial.path, limit_episodes=1) == {
        "complete": False, "completed_episodes": 1, "expected_episodes": 2,
    }
    assert not (trial.root / "manifest.json").exists()
    assert not (trial.root / "selected-settings.json").exists()


def remaining_requirement(trial):
    credit = retained_bytes(trial.root)
    assert 0 < credit < trial.required - trial.floor
    return trial.required - credit, credit


def interrupt_after_first_frame(trial, monkeypatch):
    original = sequential.save_frame

    def save_and_interrupt(path, *args, **kwargs):
        original(path, *args, **kwargs)
        if path.name == "frame_0000.npz":
            raise RuntimeError("injected stop after first retained frame")

    with monkeypatch.context() as patch:
        patch.setattr(sequential, "save_frame", save_and_interrupt)
        with pytest.raises(RuntimeError, match="stop after first retained frame"):
            selection.run(trial.root, trial.path)
    frames = list((trial.root / "frames").rglob("*.npz"))
    assert len(frames) == 1 and not list((trial.root / "cases").glob("*.json"))
    return frames[0]


def forbid_acquisition(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid or unadmitted evidence must not reach acquisition")
    monkeypatch.setattr(selection, "run_episode", forbidden)


def test_incomplete_selection_resumes_at_exact_verified_credit(trial, monkeypatch):
    start_partial(trial)
    before = fingerprints(trial.root)
    threshold, credit = remaining_requirement(trial)
    trial.capacity.free = threshold
    calls = []
    original = selection.retained_credit

    def record_credit(root, study, cases, prepared):
        result = original(root, study, cases, prepared)
        calls.append(result)
        return result

    monkeypatch.setattr(selection, "retained_credit", record_credit)
    result = selection.run(trial.root, trial.path, resume=True)
    assert calls == [credit]
    assert result["episode_count"] == 2 and result["exposure_count"] == 6
    assert selection.verify_run(trial.root) == result
    assert all(digest(trial.root / name) == value for name, value in before.items())


def test_incomplete_selection_refuses_one_byte_below_verified_credit(trial, monkeypatch):
    start_partial(trial)
    threshold, _ = remaining_requirement(trial)
    trial.capacity.free = threshold - 1
    before = fingerprints(trial.root)
    forbid_acquisition(monkeypatch)

    with pytest.raises(OSError, match="Insufficient disk"):
        selection.run(trial.root, trial.path, resume=True)
    assert fingerprints(trial.root) == before
    assert not (trial.root / "manifest.json").exists()


def test_rehashed_corrupt_partial_frame_cannot_earn_resume_credit(trial, monkeypatch):
    frame = interrupt_after_first_frame(trial, monkeypatch)
    with np.load(frame, allow_pickle=False) as data:
        image = data["image_electrons"].copy()
        record = json.loads(data["record_json"].item())
    image[0, 0] += 1.0
    record["image_sha256"] = hashlib.sha256(image.tobytes()).hexdigest()
    with frame.open("wb") as output:
        np.savez_compressed(
            output, image_electrons=image,
            record_json=np.array(json.dumps(record, sort_keys=True, allow_nan=False)),
            record_sha256=np.array(canonical_hash(record)),
        )
    # File/schema/hash-only credit would now admit this tampered checkpoint.
    trial.capacity.free, _ = remaining_requirement(trial)
    before = fingerprints(trial.root)
    forbid_acquisition(monkeypatch)

    with pytest.raises(ValueError, match="Retained image disagrees with detector replay"):
        selection.run(trial.root, trial.path, resume=True)
    assert fingerprints(trial.root) == before


def test_unexpected_file_cannot_manufacture_resume_credit(trial, monkeypatch):
    start_partial(trial)
    threshold, _ = remaining_requirement(trial)
    unexpected = trial.root / "unverified.bin"
    unexpected.write_bytes(b"unverified" * 4096)
    extra_bytes = min(unexpected.stat().st_size, unexpected.stat().st_blocks * 512)
    assert extra_bytes > 0
    # Blindly counting this file would turn an inadmissible resume into admission.
    trial.capacity.free = threshold - extra_bytes
    before = fingerprints(trial.root)
    forbid_acquisition(monkeypatch)

    with pytest.raises(ValueError, match="Unexpected file in selection study"):
        selection.run(trial.root, trial.path, resume=True)
    assert fingerprints(trial.root) == before


def test_padded_lock_cannot_manufacture_resume_credit(trial, monkeypatch):
    start_partial(trial)
    threshold, credit = remaining_requirement(trial)
    lock = trial.root / ".lock"
    lock.write_bytes(b"untrusted lock padding" * 4096)
    padding_bytes = min(lock.stat().st_size, lock.stat().st_blocks * 512)
    assert padding_bytes > 0
    study = json.loads((trial.root / "study.json").read_text())
    assert selection.retained_credit(trial.root, study, trial.cases, trial.prepared) == credit
    # Counting the lock payload as evidence would admit this underfunded resume.
    trial.capacity.free = threshold - padding_bytes
    before = fingerprints(trial.root)
    forbid_acquisition(monkeypatch)

    with pytest.raises(OSError, match="Insufficient disk"):
        selection.run(trial.root, trial.path, resume=True)
    assert fingerprints(trial.root) == before


@pytest.mark.parametrize("artifact", ["study.json", "preparation.json", "case.json", "frame.npz"])
def test_container_padding_cannot_inflate_verified_credit(trial, monkeypatch, artifact):
    if artifact == "frame.npz":
        target = interrupt_after_first_frame(trial, monkeypatch)
    else:
        start_partial(trial)
        target = (next((trial.root / "cases").glob("*.json"))
                  if artifact == "case.json" else trial.root / artifact)
    threshold, baseline_credit = remaining_requirement(trial)
    study = json.loads((trial.root / "study.json").read_text())
    assert selection.retained_credit(
        trial.root, study, trial.cases, trial.prepared) == baseline_credit
    original_size = target.stat().st_size
    # JSON permits trailing whitespace; ZIP readers permit this bounded trailing
    # padding. Use an incomplete frame prefix so no case-file hash needs rewriting.
    with target.open("ab") as output:
        output.write(b" " * 16384)
    assert target.stat().st_size == original_size + 16384
    assert selection.retained_credit(
        trial.root, study, trial.cases, trial.prepared) == baseline_credit
    padded = fingerprints(trial.root)

    trial.capacity.free = threshold - 1
    with monkeypatch.context() as patch:
        forbid_acquisition(patch)
        with pytest.raises(OSError, match="Insufficient disk"):
            selection.run(trial.root, trial.path, resume=True)
    assert fingerprints(trial.root) == padded

    trial.capacity.free = threshold
    result = selection.run(trial.root, trial.path, resume=True)
    assert result["episode_count"] == 2 and selection.verify_run(trial.root) == result
    assert all(digest(trial.root / name) == value for name, value in padded.items())


def test_next_frame_pending_buffer_earns_no_credit_and_resumes(trial, monkeypatch):
    original_replace = sequential.os.replace

    def interrupted_replace(source, destination):
        if Path(source).name == "frame_0001.pending":
            raise RuntimeError("injected stop before second frame publication")
        return original_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(sequential.os, "replace", interrupted_replace)
        with pytest.raises(RuntimeError, match="before second frame publication"):
            selection.run(trial.root, trial.path)
    frame = next((trial.root / "frames").rglob("frame_0000.npz"))
    pending = frame.with_name("frame_0001.pending")
    assert pending.stat().st_size > 0
    assert not pending.with_suffix(".npz").exists()
    threshold, credit = remaining_requirement(trial)
    study = json.loads((trial.root / "study.json").read_text())
    assert selection.retained_credit(trial.root, study, trial.cases, trial.prepared) == credit
    before = fingerprints(trial.root)
    trial.capacity.free = threshold - 1
    with pytest.raises(OSError, match="Insufficient disk"):
        selection.run(trial.root, trial.path, resume=True)
    assert fingerprints(trial.root) == before

    trial.capacity.free = threshold
    result = selection.run(trial.root, trial.path, resume=True)
    assert result["episode_count"] == 2 and selection.verify_run(trial.root) == result
    assert digest(frame) == before[str(frame.relative_to(trial.root))]
    assert not list(trial.root.rglob("*.pending"))


@pytest.mark.parametrize("artifact", [
    "preparation.json", "summary.json", "selection.json", "selected-settings.json", "manifest.json",
])
def test_interrupted_selection_publication_resumes_at_verified_credit(trial, monkeypatch, artifact):
    original_replace = sequential.os.replace

    def interrupted_replace(source, destination):
        if Path(source).name == artifact + ".pending" and Path(destination).name == artifact:
            raise RuntimeError("injected selection publication interruption")
        return original_replace(source, destination)

    # All publication helpers use the shared os.replace. Interrupt only the named
    # final rename, retaining the real writer's fully flushed pending payload.
    with monkeypatch.context() as patch:
        patch.setattr(sequential.os, "replace", interrupted_replace)
        with pytest.raises(RuntimeError, match="selection publication interruption"):
            selection.run(trial.root, trial.path)
    pending = trial.root / (artifact + ".pending")
    assert pending.stat().st_size > 0 and not (trial.root / artifact).exists()
    assert not (trial.root / "manifest.json").exists()
    threshold, credit = remaining_requirement(trial)
    study = json.loads((trial.root / "study.json").read_text())
    assert selection.retained_credit(trial.root, study, trial.cases, trial.prepared) == credit
    before = fingerprints(trial.root)

    trial.capacity.free = threshold - 1
    with monkeypatch.context() as patch:
        forbid_acquisition(patch)
        with pytest.raises(OSError, match="Insufficient disk"):
            selection.run(trial.root, trial.path, resume=True)
    assert fingerprints(trial.root) == before

    trial.capacity.free = threshold
    original_acquire = selection.run_episode
    acquired = []

    def record_acquisition(root, bound_study, case):
        acquired.append(case["id"])
        return original_acquire(root, bound_study, case)

    monkeypatch.setattr(selection, "run_episode", record_acquisition)
    result = selection.run(trial.root, trial.path, resume=True)
    assert acquired == ([case["id"] for case in trial.cases]
                        if artifact == "preparation.json" else [])
    assert result["episode_count"] == 2 and selection.verify_run(trial.root) == result
    assert all(digest(trial.root / name) == value for name, value in before.items()
               if not name.endswith(".pending"))
    assert not list(trial.root.rglob("*.pending"))


def test_completed_extra_artifact_rehashed_into_manifest_is_rejected(trial, monkeypatch):
    selection.run(trial.root, trial.path)
    extra = trial.root / "extra-result.json"
    extra.write_text('{"unverified": true}\n')
    path = trial.root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"][extra.name] = digest(extra)
    path.write_text(json.dumps(manifest))
    before = fingerprints(trial.root)
    trial.capacity.free = 0
    forbid_acquisition(monkeypatch)

    with pytest.raises(ValueError, match="Unexpected file in completed selection study"):
        selection.run(trial.root, trial.path, resume=True)
    assert fingerprints(trial.root) == before
