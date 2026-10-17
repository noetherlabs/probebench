from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from probebench import cohorts, selection, sequential, study as storage
from probebench.cohorts import generate_cohort
from probebench.qualification import canonical_hash
from probebench.run import digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    value = json.loads((ROOT / "configs/classical_selection_v1.json").read_text())
    cohort = value["cohort"]
    cohort["seed_hex"] = "11" * 32
    cohort["specification"].update(cohort_id="selection_unit_fixture",
        optics={"pupil_pixels": 12, "oversampling": 1, "wavelength_nm": 550.},
        strata=[{"id": "low", "rms_min_nm": 10., "rms_max_nm": 25., "base_count": 1}])
    cohort["exclusions"]["optics"] = deepcopy(cohort["specification"]["optics"])
    cohort["exclusions"]["wavefronts"] = cohort["exclusions"]["wavefronts"][:3]
    value.update(excluded_cohorts=[], replicates=1, photons_per_exposure=[2000],
                 exposure_budgets=[3], operating_reserve_bytes=0)
    value["candidates"] = [c for c in value["candidates"]
                            if c["id"] in {"random_amp_020", "random_amp_080"}]
    return value


def write_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def fingerprints(root):
    return {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()}


def test_case_identity_and_streams_ignore_order_candidate_and_budget(config):
    config.update(photons_per_exposure=[2000,20000], exposure_budgets=[3,7], replicates=2)
    prepared = selection.prepare(config)
    first = selection.cases_for(config, prepared)
    config["candidates"].reverse()
    config["photons_per_exposure"].reverse()
    config["exposure_budgets"].reverse()
    assert selection.cases_for(config) == first
    assert len({c["id"] for c in first}) == len(first) == 32
    streams = {}
    for case in first:
        key = tuple(case[k] for k in ("base_wavefront_id", "member", "replicate", "photons_per_exposure"))
        seeds = (case["detector_seed"], case["policy_seed"])
        if key in streams:
            assert streams[key] == seeds
        streams[key] = seeds
        assert seeds[0] != seeds[1]
    assert len(set(streams.values())) == len(streams) == 8


def test_candidate_policy_is_bound_and_actual_commands_scale(config, tmp_path):
    root = tmp_path / "study"
    selection.run(root, write_config(tmp_path, config))
    study = json.loads((root / "study.json").read_text())
    cases = selection.cases_for(config)
    paired = [c for c in cases if c["member"] == "primary"]
    assert len(paired) == 2
    commands = []
    bindings = []
    for case in paired:
        derived = selection.episode_study(study, case)
        assert derived["selection_study_sha256"] == canonical_hash(study)
        bindings.append(canonical_hash(derived))
        record = selection.read_selection_episode(root, study, case)
        commands.append(record["result"]["trajectory"][1]["applied_opd_nm"])
    assert bindings[0] != bindings[1]
    np.testing.assert_array_equal(np.array(commands[0]) * 4, commands[1])
    chosen = json.loads((root / "selected-settings.json").read_text())
    assert set(chosen["families"]) == {"random_probe_v2"}
    assert chosen["study_sha256"] == canonical_hash(study)
    assert chosen["selection_sha256"] == canonical_hash(json.loads((root / "selection.json").read_text()))


def test_partial_and_completed_resume_do_not_reexecute_completed_cases(config, tmp_path, monkeypatch):
    root = tmp_path / "study"
    path = write_config(tmp_path, config)
    partial = selection.run(root, path, limit_episodes=1)
    assert partial == {"complete":False,"completed_episodes":1,"expected_episodes":4}
    assert not (root / "selected-settings.json").exists()
    first = next((root / "cases").glob("*.json"))
    original = first.read_bytes()
    selection.run(root, path, resume=True)
    assert first.read_bytes() == original
    before = fingerprints(root)
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed episodes cannot be acquired again")
    monkeypatch.setattr(selection, "run_episode", forbidden)
    selection.run(root, path, resume=True)
    assert fingerprints(root) == before


def test_interrupted_frame_reuses_checkpoint_and_candidate(config, tmp_path, monkeypatch):
    root = tmp_path / "study"
    path = write_config(tmp_path, config)
    real = sequential.ClassicalPolicy.command
    def interrupt(self, history):
        if len(history) == 1:
            raise RuntimeError("infrastructure interruption")
        return real(self, history)
    monkeypatch.setattr(sequential.ClassicalPolicy, "command", interrupt)
    with pytest.raises(RuntimeError, match="infrastructure interruption"):
        selection.run(root, path)
    frame = next((root / "frames").rglob("frame_0000.npz"))
    before = frame.read_bytes()
    assert not (root / "selected-settings.json").exists()
    monkeypatch.setattr(sequential.ClassicalPolicy, "command", real)
    selection.run(root, path, resume=True)
    assert frame.read_bytes() == before


def test_numerical_failure_and_nonconvergence_remain_scored(config, tmp_path, monkeypatch):
    config["candidates"] = [{"id":"fixed", "family":"fixed_diversity_mle_v1",
                              "parameters":{"diversity_probe_nm":40.}}]
    from probebench import budget_policies
    def failed(*args, **kwargs):
        raise FloatingPointError("injected solver failure")
    monkeypatch.setattr(budget_policies, "estimate_phase_mle", failed)
    root = tmp_path / "study"
    selection.run(root, write_config(tmp_path, config))
    rows = json.loads((root / "summary.json").read_text())
    assert len(rows) == 2
    assert all(r["outcome_status"] == "policy_failure" and r["failed_fits"] == 1 for r in rows)
    result = json.loads((root / "selection.json").read_text())
    assert result["candidates"][0]["failure_count"] == 2
    expected = np.mean([r["residual_wfe_nm"] for r in rows])
    assert result["candidates"][0]["loss_nm"] == pytest.approx(expected)


def test_saved_mle_replay_does_not_reoptimize(config, tmp_path, monkeypatch):
    config["candidates"] = [{"id":"adaptive", "family":"phase_diversity_mle_v1",
                              "parameters":{"diversity_probe_nm":40.}}]
    config["common_policy"]["max_mle_iterations"] = 2
    config["exposure_budgets"] = [13]
    root = tmp_path / "study"
    path = write_config(tmp_path, config)
    selection.run(root, path)
    from probebench import budget_policies
    def forbidden(*args, **kwargs):
        raise AssertionError("Retained fit must be restored")
    monkeypatch.setattr(budget_policies, "estimate_phase_mle", forbidden)
    selection.verify_run(root)
    selection.run(root, path, resume=True)
    rows = json.loads((root / "summary.json").read_text())
    assert all(r["unconverged_fits"] > 0 and r["outcome_status"] == "completed" for r in rows)


def test_all_declared_candidates_complete_one_and_three_exposure_cells(config, tmp_path):
    declared = json.loads((ROOT / "configs/classical_selection_v1.json").read_text())
    config["candidates"] = declared["candidates"]
    config["exposure_budgets"] = [1, 3]
    config["common_policy"]["max_mle_iterations"] = 2
    root = tmp_path / "study"
    manifest = selection.run(root, write_config(tmp_path, config))
    assert manifest["episode_count"] == 56
    assert manifest["exposure_count"] == 112
    chosen = json.loads((root / "selection.json").read_text())
    assert len(chosen["candidates"]) == 14
    assert set(chosen["selected_candidates"]) == set(selection.METHODS)
    assert all(row["case_count"] == 4 for row in chosen["candidates"])
    study = json.loads((root / "study.json").read_text())
    for case in selection.cases_for(config):
        if case["exposure_budget"] == 1:
            result = selection.read_selection_episode(root, study, case)["result"]
            assert result["fit_records"] == []
            assert result["final_applied_opd_nm"] == [0.]*6


@pytest.mark.parametrize("name", ["selected-settings.json", "selection.json", "summary.json"])
def test_rehashed_derived_results_cannot_change_winner_or_loss(config, tmp_path, name):
    root = tmp_path / "study"
    selection.run(root, write_config(tmp_path, config))
    path = root / name
    data = json.loads(path.read_text())
    if name == "selected-settings.json":
        data["families"]["random_probe_v2"]["policy"]["random_amplitude_nm"] = 199
    elif name == "selection.json":
        data["candidates"][0]["loss_nm"] += 1
    else:
        data[0]["residual_wfe_nm"] += 1
    path.write_text(json.dumps(data))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"][name] = digest(path)
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Selection artifact disagrees"):
        selection.verify_run(root)


def test_full_study_admission_not_chunk_and_missing_evidence_not_repaired(config, tmp_path, monkeypatch):
    path = write_config(tmp_path, config)
    estimate = selection.storage_estimate(config)
    monkeypatch.setattr(storage.shutil, "disk_usage", lambda path:SimpleNamespace(free=estimate["required_free_bytes"]-1))
    with pytest.raises(OSError, match="Insufficient disk"):
        selection.run(tmp_path / "refused", path, limit_episodes=1)
    assert not (tmp_path / "refused").exists()
    monkeypatch.undo()
    root = tmp_path / "study"
    selection.run(root, path)
    source = root / "source" / "src" / "probebench" / "selection.py"
    source.unlink()
    before = fingerprints(root)
    with pytest.raises(ValueError, match="Missing or unlisted"):
        selection.run(root, path, resume=True)
    assert fingerprints(root) == before and not source.exists()


def test_composite_exclusions_reject_overlap_and_allow_repeated_exclusion_ids(config):
    public = deepcopy(config["cohort"])
    public["specification"]["cohort_id"] = "public_source"
    public["specification"]["split"] = "development"
    second = deepcopy(public)
    second["seed_hex"] = "02" * 32
    config["excluded_cohorts"] = [{"configuration":c,
        "clusters_sha256":canonical_hash(generate_cohort(c)["clusters"])} for c in (public, second)]
    prepared = selection.prepare(config)
    assert len(prepared["exclusion_cohorts"]) == 2
    assert prepared["exclusion_cohorts"][0]["cohort_sha256"] != prepared["exclusion_cohorts"][1]["cohort_sha256"]
    config["excluded_cohorts"].append({"configuration":deepcopy(config["cohort"]),
        "clusters_sha256":canonical_hash(generate_cohort(config["cohort"])["clusters"])})
    with pytest.raises(ValueError, match="reuses a previously exposed base identity"):
        selection.prepare(config)


@pytest.fixture
def forbid_selection_admission(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Selection admission or acquisition must not start")
    monkeypatch.setattr(selection, "study_directory", forbidden)
    monkeypatch.setattr(selection, "run_episode", forbidden)


@pytest.mark.parametrize("changed_field", ["seed", "split"])
def test_selection_rejects_reused_base_identity_before_admission(
        config, tmp_path, forbid_selection_admission, changed_field):
    excluded = deepcopy(config["cohort"])
    if changed_field == "seed":
        excluded["seed_hex"] = "22" * 32
    else:
        excluded["specification"]["split"] = "development"
    current = generate_cohort(config["cohort"])
    previous = generate_cohort(excluded)
    current_base, previous_base = current["clusters"][0], previous["clusters"][0]
    assert current_base["base_wavefront_id"] == previous_base["base_wavefront_id"]
    distance = cohorts.orbit_distances(
        current_base["members"][0]["truth_opd_nm"],
        [previous_base["members"][0]["truth_opd_nm"]],
        np.asarray(current["gram_nm2_per_nm2"]))[0]
    assert distance > config["cohort"]["specification"]["overlap_tolerance_nm"]
    config["excluded_cohorts"] = [{"configuration": excluded,
        "clusters_sha256": canonical_hash(previous["clusters"])}]
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="reuses a previously exposed base identity"):
        selection.run(output, write_config(tmp_path, config))
    assert not output.exists()


@pytest.mark.parametrize("conjugated", [False, True], ids=["direct", "conjugate"])
def test_selection_rejects_relabelled_orbit_overlap_before_admission(
        config, tmp_path, monkeypatch, forbid_selection_admission, conjugated):
    current = generate_cohort(config["cohort"])
    excluded = deepcopy(config["cohort"])
    excluded["specification"].update(cohort_id="previous_source", split="development")
    previous = deepcopy(current)
    previous["configuration_sha256"] = canonical_hash(excluded)
    previous_base = previous["clusters"][0]
    previous_base["base_wavefront_id"] = "previous_source:low:000000"
    if conjugated:
        vector = cohorts.conjugate(previous_base["members"][0]["truth_opd_nm"])
        previous_base["members"] = cohorts.cluster_members(
            vector, np.asarray(current["gram_nm2_per_nm2"]),
            excluded["specification"]["overlap_tolerance_nm"])
    assert previous_base["base_wavefront_id"] != current["clusters"][0]["base_wavefront_id"]
    config["excluded_cohorts"] = [{"configuration": excluded,
        "clusters_sha256": canonical_hash(previous["clusters"])}]

    def generated(configuration):
        if configuration == excluded:
            return deepcopy(previous)
        return generate_cohort(configuration)

    monkeypatch.setattr(selection, "generate_cohort", generated)
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="overlaps previously exposed wavefronts"):
        selection.run(output, write_config(tmp_path, config))
    assert not output.exists()


def test_selection_reserved_split_refused_before_admission(
        config, tmp_path, forbid_selection_admission):
    config["cohort"]["specification"]["split"] = "reserved"
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="requires an explicit selection cohort"):
        selection.run(output, write_config(tmp_path, config))
    assert not output.exists()


@pytest.mark.parametrize("mutate", [
    lambda c:c.update(schema=True),
    lambda c:c.update(purpose="reserved_evaluation"),
    lambda c:c["cohort"]["specification"].update(split="development"),
    lambda c:c.update(replicates=True),
    lambda c:c.update(exposure_budgets=[3,3]),
    lambda c:c.update(exposure_budgets=[1000]),
    lambda c:c.update(photons_per_exposure=[True]),
    lambda c:c.update(detector_seed_hex="x"),
    lambda c:c["common_policy"].update(random_amplitude_nm=True),
    lambda c:c["candidates"][0]["parameters"].update(coordinate_step_nm=5.),
    lambda c:c["candidates"][0]["parameters"].update(random_amplitude_nm=201.),
    lambda c:c["candidates"][1]["parameters"].update(random_amplitude_nm=20),
    lambda c:c["candidates"][0].update(id="../unsafe"),
])
def test_invalid_selection_contract_rejected_before_output(config, tmp_path, mutate):
    mutate(config)
    with pytest.raises(ValueError):
        selection.run(tmp_path / "study", write_config(tmp_path, config))
    assert not (tmp_path / "study").exists()


def test_configuration_change_cannot_resume_and_selection_remains_complete(config, tmp_path):
    root = tmp_path / "study"
    path = write_config(tmp_path, config)
    selection.run(root, path)
    before = fingerprints(root)
    config["candidates"][0]["parameters"]["random_amplitude_nm"] = 21.
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Source, configuration or runtime changed"):
        selection.run(root, path, resume=True)
    assert fingerprints(root) == before


def test_wrong_policy_acquisition_cannot_publish_selection(config, tmp_path, monkeypatch):
    real = selection.run_episode
    class WrongPolicy(selection.ClassicalPolicy):
        def command(self, history):
            return np.ones(6) if not history else super().command(history)
    def incorrect_acquisition(*args, **kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(sequential, "ClassicalPolicy", WrongPolicy)
            return real(*args, **kwargs)
    monkeypatch.setattr(selection, "run_episode", incorrect_acquisition)
    root = tmp_path / "study"
    with pytest.raises(ValueError, match="Retained command disagrees"):
        selection.run(root, write_config(tmp_path, config))
    assert list((root / "cases").glob("*.json"))
    for filename in ("selection.json", "selected-settings.json", "manifest.json"):
        assert not (root / filename).exists()


def test_changed_exclusion_fingerprint_is_rejected(config):
    public = deepcopy(config["cohort"])
    public["specification"].update(cohort_id="other_source",split="development")
    config["excluded_cohorts"] = [{"configuration":public,"clusters_sha256":"0"*64}]
    with pytest.raises(ValueError, match="fingerprint changed"):
        selection.prepare(config)


def replace_case_and_manifest(root, case, record):
    record.pop("record_sha256")
    record["record_sha256"] = canonical_hash(record)
    name = f'cases/{case["id"]}.json'
    (root / name).write_text(json.dumps(record))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"][name] = digest(root / name)
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_rehashed_spurious_fit_on_non_estimator_is_rejected(config, tmp_path):
    from probebench.budget_policies import visible_history_digest
    root = tmp_path / "study"
    selection.run(root, write_config(tmp_path, config))
    study = json.loads((root / "study.json").read_text())
    case = selection.cases_for(config)[0]
    derived = selection.episode_study(study, case)
    record = selection.read_selection_episode(root, study, case)
    _, _, history = selection.replay_frames(root, derived, case, case["exposure_budget"])
    record["result"]["fit_records"] = [{"observations_used":3,
        "visible_history_sha256":visible_history_digest(history),
        "estimated_opd_nm":[0.]*6, "fit_status":"finite", "fit_elapsed_seconds":0.,
        "optimizer_success":True}]
    replace_case_and_manifest(root, case, record)
    with pytest.raises(ValueError, match="declared policy schedule"):
        selection.verify_run(root)


@pytest.mark.parametrize("damage", ["command", "fallback", "reason", "success", "duration"])
def test_rehashed_numerical_failure_semantics_are_verified(config, tmp_path, monkeypatch, damage):
    from probebench import budget_policies
    config["candidates"] = [{"id":"fixed", "family":"fixed_diversity_mle_v1",
                              "parameters":{"diversity_probe_nm":40.}}]
    def failed(*args, **kwargs):
        raise FloatingPointError("injected solver failure")
    monkeypatch.setattr(budget_policies, "estimate_phase_mle", failed)
    root = tmp_path / "study"
    selection.run(root, write_config(tmp_path, config))
    study = json.loads((root / "study.json").read_text())
    case = selection.cases_for(config)[0]
    derived = selection.episode_study(study, case)
    record = selection.read_selection_episode(root, study, case)
    fit = record["result"]["fit_records"][0]
    if damage == "command":
        # Keep the physical metrics internally consistent with a different command;
        # the declared fallback rule, rather than a stale metric, must reject it.
        episode, _, _ = selection.replay_frames(root, derived, case, case["exposure_budget"])
        changed = np.array(fit["estimated_opd_nm"])
        changed[1] += 1
        fit["estimated_opd_nm"] = changed.tolist()
        record["result"].update(episode.submit(changed))
    elif damage == "fallback":
        fit["fallback"] = "arbitrary_command"
    elif damage == "reason":
        fit["reason"] = " "
    elif damage == "success":
        fit["optimizer_success"] = True
    else:
        fit["fit_elapsed_seconds"] = -1.
    replace_case_and_manifest(root, case, record)
    message = "fit duration" if damage == "duration" else "camera-score fallback"
    with pytest.raises(ValueError, match=message):
        selection.verify_run(root)
