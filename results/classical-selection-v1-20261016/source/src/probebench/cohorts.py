"""Physical-RMS wavefront sampling with conjugate-aware split validation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hmac
import json
from pathlib import Path
import re

import numpy as np

from .optics import MODES, OpticalModel, OpticsConfig, coefficients
from .qualification import canonical_hash, publish_json
from .run import FIXTURES
from .study import publish_manifest, study_directory, verify_study

PURPOSE = "wavefront_cohort_v1"
SAMPLER = "finite_pupil_isotropic_uniform_rms_pcg64_v1"
CONJUGATE_SIGNS = np.array([-1., -1., -1., 1., 1., -1.])
CONJUGATE_SIGNS.setflags(write=False)
MAX_BASES = 4096
IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"Unexpected {name} fields")


def _positive(value, name, *, zero=False):
    if (type(value) not in (int, float) or not np.isfinite(value)
            or (value < 0 if zero else value <= 0)):
        raise ValueError(f"Invalid finite {name}")


def _identifier(value):
    if type(value) is not str or IDENTIFIER.fullmatch(value) is None:
        raise ValueError("Identifiers must use lowercase ASCII letters, digits and underscores")
    return value


def pupil_gram(model):
    """Gram matrix of the actual piston-removed finite-pupil mode samples."""
    basis = model.basis[:, model.pupil].copy()
    basis -= basis.mean(axis=1, keepdims=True)
    return basis @ basis.T / basis.shape[1]


def conjugate(vector):
    """Coefficients of -w(-r), whose focal intensity equals that of w(r)."""
    return coefficients(vector) * CONJUGATE_SIGNS


def physical_rms(vector, gram):
    vector = coefficients(vector)
    return float(np.sqrt(vector @ gram @ vector))


def orbit_distances(vector, others, gram):
    """Distances in nm to each conjugate orbit; no intensity uniqueness claim."""
    vector = coefficients(vector)
    others = np.asarray(others, dtype=np.float64)
    if others.ndim != 2 or others.shape[1] != len(MODES) or not np.isfinite(others).all():
        raise ValueError("Expected finite wavefront matrix with six columns")
    direct = others - vector
    paired = others * CONJUGATE_SIGNS - vector
    squared = np.minimum(np.einsum("ni,ij,nj->n", direct, gram, direct),
                         np.einsum("ni,ij,nj->n", paired, gram, paired))
    return np.sqrt(np.maximum(squared, 0))


def cluster_members(vector, gram, tolerance_nm):
    """Collapse zero or numerically self-conjugate fixtures to one member."""
    _positive(tolerance_nm, "overlap tolerance")
    primary = coefficients(vector)
    partner = conjugate(primary)
    result = [{"member": "primary", "truth_opd_nm": primary.tolist()}]
    if physical_rms(primary - partner, gram) > tolerance_nm:
        result.append({"member": "conjugate", "truth_opd_nm": partner.tolist()})
    return result


def stream_seed(seed_hex, domain, identity):
    """HMAC-derived 256-bit PCG64 seed; caller supplies an explicit random domain."""
    if type(seed_hex) is not str or re.fullmatch(r"[0-9a-f]{64}", seed_hex) is None:
        raise ValueError("Seed must contain exactly 64 lowercase hexadecimal characters")
    if domain not in {"wavefront", "detector", "policy", "bootstrap"}:
        raise ValueError("Unknown random stream domain")
    payload = json.dumps([SAMPLER, domain, identity], sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return int.from_bytes(hmac.digest(bytes.fromhex(seed_hex), payload, "sha256"), "big")


def validate_specification(spec):
    _keys(spec, {"schema", "cohort_id", "split", "optics", "strata",
                 "command_limit_nm", "overlap_tolerance_nm"}, "cohort specification")
    if type(spec["schema"]) is not int or spec["schema"] != 1:
        raise ValueError("Unsupported cohort schema")
    _identifier(spec["cohort_id"])
    if spec["split"] not in {"development", "selection"}:
        raise ValueError("Only development and selection cohorts are supported")
    _keys(spec["optics"], {"pupil_pixels", "oversampling", "wavelength_nm"}, "optics")
    _positive(spec["optics"]["wavelength_nm"], "wavelength")
    optics = OpticsConfig(**spec["optics"])
    if optics.pupil_pixels > 256 or optics.pupil_pixels * optics.oversampling > 1024:
        raise ValueError("Cohort generation requires at most 256 pupil and 1024 focal pixels")
    for key in ("command_limit_nm", "overlap_tolerance_nm"):
        _positive(spec[key], key)
    strata = spec["strata"]
    if type(strata) is not list or not strata or len(strata) > MAX_BASES:
        raise ValueError("A bounded nonempty stratum list is required")
    names, count, intervals = set(), 0, []
    for stratum in strata:
        _keys(stratum, {"id", "rms_min_nm", "rms_max_nm", "base_count"}, "RMS stratum")
        name = _identifier(stratum["id"])
        _positive(stratum["rms_min_nm"], "minimum RMS", zero=True)
        _positive(stratum["rms_max_nm"], "maximum RMS")
        if name in names or stratum["rms_min_nm"] >= stratum["rms_max_nm"]:
            raise ValueError("RMS strata require unique names and increasing bounds")
        if type(stratum["base_count"]) is not int or stratum["base_count"] < 1:
            raise ValueError("Each stratum requires a positive integer base count")
        names.add(name)
        count += stratum["base_count"]
        intervals.append((stratum["rms_min_nm"], stratum["rms_max_nm"]))
    if count > MAX_BASES:
        raise ValueError(f"At most {MAX_BASES} independent bases are supported")
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("RMS strata must not overlap")
    model = OpticalModel(optics)
    gram = pupil_gram(model)
    envelope = max(hi for _, hi in intervals) * np.sqrt(np.diag(np.linalg.inv(gram)).max())
    if envelope > spec["command_limit_nm"]:
        raise ValueError("RMS support exceeds the all-directions correctable envelope")
    return model, gram


def validate_registry(registry, model):
    _keys(registry, {"schema", "optics", "wavefronts"}, "exclusion registry")
    if type(registry["schema"]) is not int or registry["schema"] != 1:
        raise ValueError("Unsupported exclusion schema")
    if registry["optics"] != asdict(model.config):
        raise ValueError("Exclusion registry uses different optics")
    rows = registry["wavefronts"]
    if type(rows) is not list or not rows or len(rows) > MAX_BASES:
        raise ValueError("A bounded nonempty exclusion registry is required")
    identifiers, vectors = set(), []
    for row in rows:
        _keys(row, {"id", "truth_opd_nm", "sources"}, "excluded wavefront")
        identity = _identifier(row["id"])
        if identity in identifiers:
            raise ValueError("Duplicate exclusion identity")
        identifiers.add(identity)
        if (type(row["sources"]) is not list or not row["sources"]
                or any(type(s) is not str or not s for s in row["sources"])):
            raise ValueError("Exclusions require provenance references")
        raw = row["truth_opd_nm"]
        if type(raw) is not list or any(type(v) not in (int, float) for v in raw):
            raise ValueError("Exclusion coefficients must be a numeric JSON array")
        vectors.append(coefficients(raw))
    # A caller may add earlier cohorts, but cannot omit the exposed original fixtures.
    # Coverage is exact: an approximately equal representative shifts its exclusion
    # ball and can admit a near-neighbour of the true original fixture.
    for _, fixture in FIXTURES:
        if not any(np.array_equal(fixture, v) or np.array_equal(conjugate(fixture), v)
                   for v in vectors):
            raise ValueError("Exclusion registry omits an original development fixture")
    return np.asarray(vectors)


def sample_base(spec, seed_hex, stratum_id, base_index, gram):
    """One base draw, independent of stratum order or generation chunk boundaries."""
    matches = [s for s in spec["strata"] if s["id"] == stratum_id]
    if (len(matches) != 1 or type(base_index) is not int
            or not 0 <= base_index < matches[0]["base_count"]):
        raise ValueError("Unknown stratum or base index")
    stratum = matches[0]
    identity = {"cohort_id": spec["cohort_id"], "split": spec["split"],
                "stratum_id": stratum_id, "base_index": base_index}
    # Radius and angular draws have separate identities, so no rejection can couple them.
    radial = np.random.Generator(np.random.PCG64(stream_seed(seed_hex, "wavefront",
                                {**identity, "component": "radius"})))
    angular = np.random.Generator(np.random.PCG64(stream_seed(seed_hex, "wavefront",
                                 {**identity, "component": "direction"})))
    radius = float(radial.uniform(stratum["rms_min_nm"], stratum["rms_max_nm"]))
    direction = angular.normal(size=len(MODES))
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Degenerate Gaussian direction; no resampling permitted")
    vector = radius * np.linalg.solve(np.linalg.cholesky(gram).T, direction / norm)
    if not np.isfinite(vector).all() or np.max(np.abs(vector)) > spec["command_limit_nm"]:
        raise ValueError("Sample exceeds the declared correctable envelope")
    return {"base_wavefront_id": f'{spec["cohort_id"]}:{stratum_id}:{base_index:06d}',
            "stratum_id": stratum_id, "base_index": base_index,
            "sampled_rms_nm": radius, "physical_rms_nm": physical_rms(vector, gram),
            "members": cluster_members(vector, gram, spec["overlap_tolerance_nm"])}


def generate_cohort(configuration):
    _keys(configuration, {"specification", "seed_hex", "exclusions",
                          "operating_reserve_bytes"}, "cohort configuration")
    spec = configuration["specification"]
    model, gram = validate_specification(spec)
    stream_seed(configuration["seed_hex"], "wavefront", {})
    reserve = configuration["operating_reserve_bytes"]
    if type(reserve) is not int or reserve < 0:
        raise ValueError("Operating reserve must be a nonnegative integer")
    tolerance = spec["overlap_tolerance_nm"]
    excluded = validate_registry(configuration["exclusions"], model)
    vectors, records = [], []
    for stratum in sorted(spec["strata"], key=lambda s: s["id"]):
        for index in range(stratum["base_count"]):
            record = sample_base(spec, configuration["seed_hex"], stratum["id"], index, gram)
            vector = record["members"][0]["truth_opd_nm"]
            if orbit_distances(vector, excluded, gram).min() <= tolerance:
                raise ValueError(f'Wavefront overlaps an exclusion: {record["base_wavefront_id"]}')
            if vectors and orbit_distances(vector, vectors, gram).min() <= tolerance:
                raise ValueError(f'Duplicate conjugate orbit: {record["base_wavefront_id"]}')
            vectors.append(vector)
            records.append(record)
    return {"schema": 1, "sampler": SAMPLER, "configuration_sha256": canonical_hash(configuration),
            "mode_order": list(MODES), "gram_nm2_per_nm2": gram.tolist(),
            "base_count": len(records), "member_count": sum(len(r["members"]) for r in records),
            "clusters": records}


def assert_disjoint(configurations):
    """Regenerate and compare all base orbits, irrespective of labels or seeds.

    This checks the supplied configurations, not completeness of the caller's list.
    An exact-content registry remains necessary for all previously exposed fixtures.
    """
    if not isinstance(configurations, (list, tuple)) or len(configurations) < 2:
        raise ValueError("At least two cohort configurations are required")
    cohorts = [generate_cohort(c) for c in configurations]
    optics = configurations[0]["specification"]["optics"]
    if any(c["specification"]["optics"] != optics for c in configurations):
        raise ValueError("Cannot compare cohorts with different optical bases")
    gram = pupil_gram(OpticalModel(OpticsConfig(**optics)))
    tolerance = max(c["specification"]["overlap_tolerance_nm"] for c in configurations)
    seen, identities = [], set()
    for cohort in cohorts:
        for cluster in cohort["clusters"]:
            identity = cluster["base_wavefront_id"]
            vector = cluster["members"][0]["truth_opd_nm"]
            if identity in identities:
                raise ValueError("Duplicate base identity across cohorts")
            if seen and orbit_distances(vector, seen, gram).min() <= tolerance:
                raise ValueError(f"Cohort overlap modulo conjugacy: {identity}")
            seen.append(vector)
            identities.add(identity)
    return {"cohorts": len(cohorts), "independent_bases": len(seen),
            "overlap_tolerance_nm": tolerance}


def verify_cohort(root):
    root = Path(root)
    manifest, study = verify_study(root, PURPOSE)
    expected = generate_cohort(study["configuration"])
    if json.loads((root / "cohort.json").read_text()) != expected:
        raise ValueError("Cohort does not replay from its specification and seed")
    if (manifest.get("base_count") != expected["base_count"]
            or manifest.get("member_count") != expected["member_count"]):
        raise ValueError("Cohort manifest counts disagree")
    return manifest


def run(root, config_path, resume=False):
    configuration = json.loads(Path(config_path).read_text())
    cohort = generate_cohort(configuration)  # Reject unsupported splits before any output.
    root = Path(root)
    # A completed archive must be verified before study_directory could restore a
    # missing source snapshot. Completed resumption is read-only, never a repair.
    if (root / "manifest.json").exists():
        verify_cohort(root)
    with study_directory(root, configuration, len(json.dumps(cohort).encode()), resume) as study:
        if (root / "manifest.json").exists():
            return verify_cohort(root)
        path = root / "cohort.json"
        if path.exists():
            if json.loads(path.read_text()) != cohort:
                raise ValueError("Saved cohort differs from deterministic generation")
        else:
            publish_json(path, cohort)
        publish_manifest(root, study, PURPOSE, base_count=cohort["base_count"],
                         member_count=cohort["member_count"])
    return verify_cohort(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.config or args.output or args.resume:
            parser.error("--verify cannot be combined with generation options")
        result = verify_cohort(args.verify)
    else:
        if not args.config or not args.output:
            parser.error("--config and --output are required")
        result = run(args.output, args.config, args.resume)
    print(json.dumps({"purpose": result["purpose"], "base_count": result["base_count"],
                      "member_count": result["member_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
