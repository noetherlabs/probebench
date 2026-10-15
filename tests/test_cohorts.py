from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from probebench import cohorts
from probebench.optics import OpticalModel, OpticsConfig
from probebench.qualification import canonical_hash
from probebench.run import digest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    value = json.loads((ROOT / "configs/cohort_development.json").read_text())
    value["operating_reserve_bytes"] = 0
    return value


@pytest.mark.parametrize("n", [12, 13, 32, 63])
def test_physical_gram_and_conjugacy_against_direct_pupil_and_fft(n):
    model = OpticalModel(OpticsConfig(pupil_pixels=n))
    gram = cohorts.pupil_gram(model)
    vector = np.array([31., -12., 8., 19., -7., 15.])
    partner = cohorts.conjugate(vector)
    np.testing.assert_array_equal(cohorts.conjugate(partner), vector)
    np.testing.assert_allclose(gram * np.outer(cohorts.CONJUGATE_SIGNS,
                                            cohorts.CONJUGATE_SIGNS), gram, atol=1e-15)
    direct_rms = np.std(model.wavefront_nm(vector)[model.pupil])
    assert cohorts.physical_rms(vector, gram) == pytest.approx(direct_rms, abs=1e-12)
    np.testing.assert_allclose(model.intensity(vector), model.intensity(partner), atol=1e-16)
    probe = np.array([80., 0, 0, 0, 0, 0])
    assert np.linalg.norm(model.intensity(vector-probe)-model.intensity(partner-probe)) > .01


def test_radius_direction_distribution_and_correctable_support(config):
    spec = config["specification"]
    spec["strata"] = [{"id": "shell", "rms_min_nm": 20., "rms_max_nm": 80.,
                       "base_count": 4096}]
    model, gram = cohorts.validate_specification(spec)
    # Independent direct pupil geometry checks every draw; moment checks use a fixed
    # public test seed and broad tolerances, not a test of scientific performance.
    records = [cohorts.sample_base(spec, config["seed_hex"], "shell", i, gram)
               for i in range(4096)]
    vectors = np.array([r["members"][0]["truth_opd_nm"] for r in records])
    radii = np.array([r["sampled_rms_nm"] for r in records])
    direct = np.std(vectors @ model.basis[:, model.pupil], axis=1)
    np.testing.assert_allclose(direct, radii, rtol=1e-14, atol=1e-12)
    directions = vectors @ np.linalg.cholesky(gram) / radii[:, None]
    np.testing.assert_allclose((directions**2).sum(axis=1), 1, atol=2e-15)
    assert np.max(np.abs(directions.mean(axis=0))) < .025
    np.testing.assert_allclose(directions.T @ directions / len(directions),
                               np.eye(6)/6, atol=.012)
    assert 49 < radii.mean() < 51  # Uniform RMS, not shell-volume weighting.
    assert np.max(np.abs(vectors)) <= spec["command_limit_nm"]
    assert np.max(np.abs(gram - np.eye(6))) > .001


def test_sample_identities_ignore_iteration_order_and_chunking(config):
    original = cohorts.generate_cohort(config)
    reordered = deepcopy(config)
    reordered["specification"]["strata"].reverse()
    assert cohorts.generate_cohort(reordered)["clusters"] == original["clusters"]
    _, gram = cohorts.validate_specification(config["specification"])
    records = []
    for row in reversed(original["clusters"]):
        records.append(cohorts.sample_base(config["specification"], config["seed_hex"],
                                          row["stratum_id"], row["base_index"], gram))
    assert records[::-1] == original["clusters"]
    ids = [r["base_wavefront_id"] for r in original["clusters"]]
    assert len(ids) == len(set(ids)) == original["base_count"] == 12
    assert original["member_count"] == 24


def test_conjugate_singletons_and_pure_coma_sign_are_distinct():
    gram = cohorts.pupil_gram(OpticalModel(OpticsConfig()))
    for vector in ([0.]*6, [0., 0, 0, 25, -10, 0]):
        assert len(cohorts.cluster_members(vector, gram, 1e-6)) == 1
    coma = np.array([0., 0, 0, 25, -10, 0])
    assert cohorts.orbit_distances(coma, [-coma], gram)[0] > 50
    mixed = np.array([30., 12, -10, 8, -6, 15])
    assert len(cohorts.cluster_members(mixed, gram, 1e-6)) == 2
    assert cohorts.orbit_distances(mixed, [cohorts.conjugate(mixed)], gram)[0] == 0


@pytest.mark.parametrize("near,conjugated", [(False, False), (True, False),
                                           (False, True), (True, True)])
def test_exclusion_overlap_rejected_modulo_conjugacy(config, near, conjugated):
    first = cohorts.generate_cohort(config)["clusters"][0]["members"][0]["truth_opd_nm"]
    vector = cohorts.conjugate(first) if conjugated else np.array(first)
    if near:
        vector[0] += 0.5 * config["specification"]["overlap_tolerance_nm"]
    config["exclusions"]["wavefronts"].append({"id": "earlier_cohort", "truth_opd_nm": vector.tolist(),
                                               "sources": ["retained/cohort.json#/clusters/0"]})
    with pytest.raises(ValueError, match="overlaps an exclusion"):
        cohorts.generate_cohort(config)


def test_within_cohort_duplicate_is_not_silently_resampled(config, monkeypatch):
    real = cohorts.sample_base
    def repeated(spec, seed, name, index, gram):
        return real(spec, seed, name, 0, gram)
    monkeypatch.setattr(cohorts, "sample_base", repeated)
    with pytest.raises(ValueError, match="Duplicate conjugate orbit"):
        cohorts.generate_cohort(config)


def test_disjoint_configurations_and_cross_split_overlap(config, monkeypatch):
    other = deepcopy(config)
    other["specification"]["cohort_id"] = "selection_example"
    other["specification"]["split"] = "selection"
    result = cohorts.assert_disjoint([config, other])
    assert result["independent_bases"] == 24
    with pytest.raises(ValueError, match="Duplicate base identity"):
        cohorts.assert_disjoint([config, config])
    real = cohorts.sample_base
    def reuse(spec, seed, name, index, gram):
        record = real(config["specification"], seed, name, index, gram)
        record["base_wavefront_id"] = spec["cohort_id"] + ":" + name + str(index)
        if spec["split"] == "selection":
            vector = cohorts.conjugate(record["members"][0]["truth_opd_nm"])
            record["members"] = cohorts.cluster_members(vector, gram, 1e-6)
        return record
    monkeypatch.setattr(cohorts, "sample_base", reuse)
    with pytest.raises(ValueError, match="overlap modulo conjugacy"):
        cohorts.assert_disjoint([config, other])


def test_random_domains_are_deterministic_distinct_and_identity_bound():
    seed = "01" * 32
    identity = {"base": "example", "replicate": 1, "photons": 2000}
    samples = [cohorts.stream_seed(seed, domain, identity)
               for domain in ("wavefront", "detector", "policy", "bootstrap")]
    assert len(set(samples)) == 4
    assert cohorts.stream_seed(seed, "detector", dict(reversed(list(identity.items())))) == samples[1]
    for different in (identity | {"replicate": 2}, identity | {"photons": 20000}):
        assert cohorts.stream_seed(seed, "detector", different) != samples[1]
    assert cohorts.stream_seed("02"*32, "detector", identity) != samples[1]


@pytest.mark.parametrize("mutation", [
    lambda c: c.update(seed_hex="1"),
    lambda c: c.update(seed_hex="AB"*32),
    lambda c: c.update(operating_reserve_bytes=True),
    lambda c: c["specification"].update(split="reserved"),
    lambda c: c["specification"].update(schema=True),
    lambda c: c["specification"].update(command_limit_nm=50),
    lambda c: c["specification"].update(overlap_tolerance_nm=float("nan")),
    lambda c: c["specification"]["optics"].update(wavelength_nm=True),
    lambda c: c["specification"]["optics"].update(pupil_pixels=1000),
    lambda c: c["specification"]["strata"][0].update(base_count=True),
    lambda c: c["specification"]["strata"][0].update(base_count=4097),
    lambda c: c["specification"]["strata"][0].update(rms_max_nm=51),
    lambda c: c["specification"]["strata"][0].update(id="../escape"),
    lambda c: c["exclusions"]["wavefronts"].pop(0),
    lambda c: c["exclusions"]["optics"].update(pupil_pixels=64),
    lambda c: c["exclusions"]["wavefronts"][0].update(truth_opd_nm=[float("inf")]*6),
    lambda c: c["exclusions"]["wavefronts"][0].update(truth_opd_nm=["-45", 0, 0, 0, 0, 0]),
    lambda c: c["exclusions"]["wavefronts"][0].update(sources=[]),
])
def test_invalid_configuration_rejected_before_writing(config, mutation, tmp_path):
    mutation(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        cohorts.run(tmp_path / "output", path)
    assert not (tmp_path / "output").exists()


def test_registry_covers_retained_exposed_truths_and_valid_source_pointers(config):
    gram = cohorts.pupil_gram(OpticalModel(OpticsConfig()))
    rows = config["exclusions"]["wavefronts"]
    vectors = [row["truth_opd_nm"] for row in rows]
    original = json.loads((ROOT / "results/development-20261014/configuration.json").read_text())
    exposed = [fixture[1] for fixture in original["fixtures"]]
    for filename in ("noise_qualification_development.json", "sequential_calibration_development.json",
                     "measurement_schedule_development.json"):
        exposed.extend(json.loads((ROOT / "configs" / filename).read_text())["truths_opd_nm"])
    for vector in exposed:
        assert cohorts.orbit_distances(vector, vectors, gram).min() <= 1e-6
    for row in rows:
        for reference in row["sources"]:
            path, pointer = reference.split("#")
            value = json.loads((ROOT / path).read_text())
            for key in pointer.split("/")[1:]:
                value = value[int(key)] if isinstance(value, list) else value[key]
            np.testing.assert_array_equal(value, row["truth_opd_nm"])


def test_artifact_resume_verification_and_rehashed_tamper(config, tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    root = tmp_path / "cohort"
    real_publish = cohorts.publish_manifest
    def interrupt(*args, **kwargs):
        raise RuntimeError("injected interruption")
    monkeypatch.setattr(cohorts, "publish_manifest", interrupt)
    with pytest.raises(RuntimeError, match="injected interruption"):
        cohorts.run(root, config_path)
    saved = (root / "cohort.json").read_bytes()
    monkeypatch.setattr(cohorts, "publish_manifest", real_publish)
    cohorts.run(root, config_path, resume=True)
    assert saved == (root / "cohort.json").read_bytes()
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    cohorts.run(root, config_path, resume=True)
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        cohorts.run(root, config_path)
    record = json.loads((root / "cohort.json").read_text())
    record["clusters"][0]["members"][0]["truth_opd_nm"][0] += 1
    (root / "cohort.json").write_text(json.dumps(record))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"]["cohort.json"] = digest(root / "cohort.json")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not replay"):
        cohorts.verify_cohort(root)


def test_resume_rejects_configuration_source_and_runtime_changes(config, tmp_path, monkeypatch):
    from probebench import study as storage
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    root = tmp_path / "cohort"
    cohorts.run(root, path)
    config["seed_hex"] = "02" * 32
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Source, configuration or runtime changed"):
        cohorts.run(root, path, resume=True)
    study = json.loads((root / "study.json").read_text())
    config = study["configuration"]
    path.write_text(json.dumps(config))
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with monkeypatch.context() as patch:
        patch.setattr(storage.platform, "python_version", lambda: "0.0")
        with pytest.raises(ValueError, match="Source, configuration or runtime changed"):
            cohorts.run(root, path, resume=True)
    source_root, sources = storage.source_inventory()
    sources["src/probebench/cohorts.py"] = "0" * 64
    with monkeypatch.context() as patch:
        patch.setattr(storage, "source_inventory", lambda: (source_root, sources))
        with pytest.raises(ValueError, match="Source, configuration or runtime changed"):
            cohorts.run(root, path, resume=True)
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_cohort_binds_every_exclusion_reference(config):
    original = cohorts.generate_cohort(config)
    config["exclusions"]["wavefronts"][0]["sources"].append("additional/source.json#/fixture")
    updated = cohorts.generate_cohort(config)
    assert original["clusters"] == updated["clusters"]
    assert original["configuration_sha256"] != updated["configuration_sha256"]
    assert updated["configuration_sha256"] == canonical_hash(config)


def test_completed_resume_never_repairs_missing_source(config, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    root = tmp_path / "cohort"
    cohorts.run(root, path)
    deleted = root / "source" / "src" / "probebench" / "optics.py"
    deleted.unlink()
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="Missing or unlisted"):
        cohorts.run(root, path, resume=True)
    assert not deleted.exists()
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_approximate_registry_cannot_shift_original_fixture_exclusion(config):
    config["exclusions"]["wavefronts"][0]["truth_opd_nm"][0] += 0.8e-6
    with pytest.raises(ValueError, match="omits an original development fixture"):
        cohorts.generate_cohort(config)


@pytest.mark.parametrize("stage", ["study.json.pending", "source_pending", "cohort.json.pending",
                                   "manifest.json.pending"])
def test_interrupted_publication_buffers_resume(config, tmp_path, stage):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    root = tmp_path / "cohort"
    cohorts.run(root, path)
    manifest = (root / "manifest.json").read_bytes()
    (root / "manifest.json").unlink()
    if stage == "study.json.pending":
        import shutil
        shutil.rmtree(root / "source")
        (root / "study.json").unlink()
        (root / "cohort.json").unlink()
        (root / stage).write_text('{"interrupted":')
    elif stage == "source_pending":
        source = root / "source" / "src" / "probebench" / "cohorts.py"
        source.unlink()
        source.with_suffix(".py.pending").write_text("interrupted source copy")
        (root / "cohort.json").unlink()
    elif stage == "cohort.json.pending":
        (root / "cohort.json").unlink()
        (root / stage).write_text('{"interrupted":')
    else:
        (root / stage).write_text('{"interrupted":')
    cohorts.run(root, path, resume=True)
    assert (root / "manifest.json").read_bytes() == manifest
    assert not list(root.rglob("*.pending"))
