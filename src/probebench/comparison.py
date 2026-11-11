"""Native evidence for the frozen classical comparison, with public-only execution.

The private engine handles a complete declared inventory. The public acquisition
entry accepts only mixed_a development fixtures. Prospective generation and
execution remain closed until exclusion and custody admission are implemented.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat

from . import comparison_pilot as pilot
from .budget_policies import PolicyConfig
from .comparison_contract import SELECTED_SOURCE_SHA256, _selected, parse_json
from .comparison_inventory import DEVELOPMENT, case_inventory, public_fixture_inventory, _validated
from .environment import EpisodeConfig
from .optics import OpticalModel, OpticsConfig
from .run import FIXTURES, source_inventory
from .study import study_directory

PURPOSE = "classical_comparison_evidence_v1"
OPTICS = {"pupil_pixels": 32, "oversampling": 2, "wavelength_nm": 550.0}
NODE_SCHEMA = "probebench.comparison-node/1"


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(path):
    return parse_json(pilot._read_selected(path))


def _base_name(identity):
    return "base-" + _sha(identity.encode("utf-8"))


def _relative(name):
    if (type(name) is not str or not 1 <= len(name) <= 256
            or str(PurePosixPath(name)) != name or PurePosixPath(name).is_absolute()
            or any(p in {"", ".", ".."} for p in name.split("/"))):
        raise ValueError("Unsafe comparison artifact path")
    return name


def _node_scan(folder, files, children):
    """Exact bounded node scan; child nodes are verified separately, never skipped."""
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError("Comparison node is not a regular directory")
    if type(files) is not dict or type(children) is not dict or len(files)+len(children) > 512:
        raise ValueError("Comparison node inventory exceeds its bound")
    for name in (*files, *children):
        _relative(name)
    if any("/" in name for name in children) or set(files) & set(children):
        raise ValueError("Invalid comparison child inventory")
    expected_dirs = {str(parent) for name in files for parent in PurePosixPath(name).parents
                     if str(parent) != "."} | set(children)
    observed, count = set(), 0
    def onerror(error):
        raise error
    for base, dirs, names in os.walk(folder, followlinks=False, onerror=onerror):
        relative = Path(base).relative_to(folder)
        count += len(dirs)+len(names)
        if count > 2048:
            raise ValueError("Comparison node entry count exceeds its bound")
        for name in dirs[:]:
            path = Path(base)/name
            if not stat.S_ISDIR(path.lstat().st_mode):
                raise ValueError("Unsafe comparison directory")
            label = str(relative/name)
            if label not in expected_dirs:
                raise ValueError("Unexpected comparison directory")
            if relative == Path(".") and name in children:
                dirs.remove(name)
        for name in names:
            path = Path(base)/name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > pilot.MAX_FILE_BYTES:
                raise ValueError("Unsafe or oversized comparison artifact")
            observed.add(str(relative/name))
    allowed = set(files) | {"manifest.json"}
    if observed != allowed:
        raise ValueError("Missing or unlisted comparison artifact")
    if any(not (folder/name).is_dir() for name in children):
        raise ValueError("Missing comparison child")


def _seal(folder, *, kind, files, children):
    value = {"schema": NODE_SCHEMA, "kind": kind,
             "files": {name: _sha(pilot._read_selected(folder/name)) for name in sorted(files)},
             "children": {name: _sha(pilot._read_selected(folder/name/"manifest.json"))
                          for name in sorted(children)}}
    pilot._json(folder/"manifest.json", value)
    _node_scan(folder, value["files"], value["children"])
    return value


def _verify_node(folder, *, kind):
    value = _json(folder/"manifest.json")
    if (type(value) is not dict or set(value) != {"schema", "kind", "files", "children"}
            or value["schema"] != NODE_SCHEMA or value["kind"] != kind):
        raise ValueError("Unexpected comparison node schema")
    _node_scan(folder, value["files"], value["children"])
    for name, expected in value["files"].items():
        if _sha(pilot._read_selected(folder/name)) != expected:
            raise ValueError("Comparison artifact digest mismatch")
    for name, expected in value["children"].items():
        if _sha(pilot._read_selected(folder/name/"manifest.json")) != expected:
            raise ValueError("Comparison child manifest digest mismatch")
    return value


def episode_config(case):
    budget, photons = case["exposure_budget"], case["photons_per_exposure"]
    return EpisodeConfig(max_exposures=budget, photon_budget=budget*photons,
                         photons_per_exposure=photons)


def development_configuration(selected_settings_bytes, *, operating_reserve_bytes):
    # Reuse the existing exact selected-bytes and explicit reserve checks.
    pilot.configuration(selected_settings_bytes, operating_reserve_bytes=operating_reserve_bytes)
    return {"schema": 1, "purpose": PURPOSE, "mode": DEVELOPMENT,
            "inventory": public_fixture_inventory(), "optics": dict(OPTICS),
            "selected_settings_sha256": _sha(selected_settings_bytes),
            "operating_reserve_bytes": operating_reserve_bytes,
            "final_submission_exposure": False,
            "statistical_inference": "none_single_exposed_public_fixture",
            "resume": "verified_completed_case_boundaries_only"}


def _development_material(case):
    # These already exposed seeds are deliberately shared across methods and
    # budgets. A different flux changes the distribution, not the stream label.
    return {"truth_opd_nm": list(dict(FIXTURES)["mixed_a"]),
            "detector_seed": pilot.DETECTOR_SEED, "policy_seed": pilot.POLICY_SEED}


def _case_identity(case, selected, material):
    return {"case": case, "policy": selected["families"][case["method"]]["policy"],
            "driver": material}


def _verify_case(folder, case, selected, material):
    manifest = _verify_node(folder, kind="case")
    expected = {"dispatch.json"} | {"records/"+name for name in pilot.episode_files(case["exposure_budget"])}
    if set(manifest["files"]) != expected or manifest["children"]:
        raise ValueError("Changed case evidence inventory")
    identity = _case_identity(case, selected, material)
    if _json(folder/"dispatch.json") != identity:
        raise ValueError("Changed case dispatch or driver identity")
    checked = pilot.verify_episode(folder/"records", method=case["method"],
        policy=identity["policy"], truth_opd_nm=material["truth_opd_nm"],
        episode_config=episode_config(case), identity=identity)
    return checked, _sha(pilot._read_selected(folder/"manifest.json"))


def _groups(inventory):
    groups = {}
    for case in case_inventory(inventory):
        groups.setdefault(case["base_wavefront_id"], []).append(case)
    return groups


def _prefix(root, inventory, selected, material_for_case):
    """Read-only exact acquisition-prefix verification before any optical work."""
    groups, count, gap = _groups(inventory), 0, False
    expected_bases = {_base_name(base) for base in groups}
    allowed = {".lock", "study.json", "source", "selected-settings.json", "manifest.json"} | expected_bases
    if any(path.name not in allowed for path in root.iterdir()):
        raise ValueError("Stopped, interrupted or foreign comparison prefix; no retry")
    for base, cases in groups.items():
        folder = root/_base_name(base)
        if not folder.exists():
            gap = True
            continue
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Unsafe comparison base directory")
        allowed_case_names = {case["id"] for case in cases}
        if any(path.name not in allowed_case_names | {"manifest.json"} for path in folder.iterdir()):
            raise ValueError("Unexpected comparison case")
        complete = True
        for case in cases:
            path = folder/case["id"]
            if not path.exists():
                complete, gap = False, True
                continue
            if gap:
                raise ValueError("Existing case is not a declared acquisition prefix")
            if not (path/"manifest.json").exists():
                raise ValueError("Interrupted dispatch requires reconciliation; no implicit retry")
            _verify_case(path, case, selected, material_for_case(case))
            count += 1
        if (folder/"manifest.json").exists():
            if not complete:
                raise ValueError("Incomplete base has a completion manifest")
            manifest = _verify_node(folder, kind="base")
            if manifest["files"] or set(manifest["children"]) != allowed_case_names:
                raise ValueError("Changed complete base inventory")
    return count


def _execute(root, configuration, selected_settings_bytes, material_for_case, *, resume, limit_cases):
    """Shared engine; caller must supply a separately admitted driver, never a policy.

    This internal function is not an execution authorization API. The only public
    caller below supplies the fixed already exposed development material.
    """
    inventory = _validated(configuration["inventory"])
    selected = _selected(selected_settings_bytes)
    if type(limit_cases) is not int or limit_cases < 1:
        raise ValueError("A positive integer case limit is required")
    _, sources = source_inventory()
    if any(sources.get(path) != sha for path, sha in SELECTED_SOURCE_SHA256.items()):
        raise ValueError("Selected optical or policy implementation changed")
    cases = case_inventory(inventory)
    # Existing study_directory retains its exclusive lock and exact-content
    # comparison. Admit its immutable resume header before its legacy JSON read.
    if resume and (root.exists() or root.is_symlink()):
        if root.is_symlink() or not root.is_dir():
            raise ValueError("Unsafe comparison resume root")
        header = root/"study.json"
        if header.exists() or header.is_symlink():
            _json(header)
    raw_bytes = sum(case["exposure_budget"]*64*64*8 for case in cases) + len(cases)*256*1024
    with study_directory(root, configuration, raw_bytes, resume=resume) as study:
        selected_path = root/"selected-settings.json"
        if selected_path.exists():
            if pilot._read_selected(selected_path) != selected_settings_bytes:
                raise ValueError("Changed selected settings")
        else:
            pilot._write(selected_path, selected_settings_bytes)
        completed = _prefix(root, inventory, selected, material_for_case)
        if (root/"manifest.json").exists():
            manifest = _verify_node(root, kind="comparison")
            expected_files = {".lock", "study.json", "selected-settings.json"} | {
                "source/"+name for name in study["source_sha256"]}
            if (set(manifest["files"]) != expected_files or set(manifest["children"]) != {
                    _base_name(base) for base in _groups(inventory)}):
                raise ValueError("Changed complete comparison inventory")
            if completed != len(cases):
                raise ValueError("Incomplete comparison has a completion manifest")
            return {"status": "complete", "completed": completed, "required": len(cases)}
        model, dispatched = None, 0
        try:
            for base, base_cases in _groups(inventory).items():
                folder = root/_base_name(base)
                folder.mkdir(exist_ok=True)
                for case in base_cases:
                    path = folder/case["id"]
                    if path.exists():
                        continue  # The entire existing prefix was verified above.
                    if dispatched == limit_cases:
                        return {"status": "bounded_prefix", "completed": completed,
                                "required": len(cases)}
                    material = material_for_case(case)
                    identity = _case_identity(case, selected, material)
                    path.mkdir()
                    pilot._json(path/"dispatch.json", identity)
                    if model is None:
                        model = OpticalModel(OpticsConfig(**configuration["optics"]))
                    pilot._episode(path/"records", case["method"], PolicyConfig(**identity["policy"]),
                        model, episode_config(case), **material, identity=identity)
                    # Verify retained physical/nominal consistency before sealing.
                    pilot.verify_episode(path/"records", method=case["method"],
                        policy=identity["policy"], truth_opd_nm=material["truth_opd_nm"],
                        episode_config=episode_config(case), identity=identity)
                    _seal(path, kind="case", children={}, files={"dispatch.json"} |
                          {"records/"+name for name in pilot.episode_files(case["exposure_budget"])})
                    dispatched += 1
                    completed += 1
                if not (folder/"manifest.json").exists():
                    _seal(folder, kind="base", files={}, children={case["id"] for case in base_cases})
            if source_inventory()[1] != study["source_sha256"]:
                raise ValueError("Source changed during comparison acquisition")
            _seal(root, kind="comparison", children={_base_name(base) for base in _groups(inventory)},
                  files={".lock", "study.json", "selected-settings.json"} |
                  {"source/"+name for name in study["source_sha256"]})
            return {"status": "complete", "completed": completed, "required": len(cases)}
        except BaseException as error:
            # Every dispatched case and raw prefix remains. A failure is never
            # replaced or given a synthetic finite outcome to satisfy a reducer.
            pilot._json(root/"stop.json", {"status": "stopped", "completed": completed,
                        "error_type": type(error).__name__, "error": str(error)[:2048]})
            raise


def run_development(output, selected_settings_bytes, *, operating_reserve_bytes,
                    resume=False, limit_cases=36):
    """Run all six methods/cells on mixed_a only; no reserved material is accepted."""
    config = development_configuration(selected_settings_bytes,
                                       operating_reserve_bytes=operating_reserve_bytes)
    return _execute(Path(output), config, selected_settings_bytes, _development_material,
                    resume=resume, limit_cases=limit_cases)


def verify_development(output):
    """Verify all 36 cases from retained bytes, with no optical or policy callbacks."""
    root = Path(output)
    manifest = _verify_node(root, kind="comparison")
    study = _json(root/"study.json")
    selected_bytes = pilot._read_selected(root/"selected-settings.json")
    expected = development_configuration(selected_bytes,
                     operating_reserve_bytes=study["configuration"]["operating_reserve_bytes"])
    if study["configuration"] != expected:
        raise ValueError("Changed development comparison configuration")
    source_hashes = study["source_sha256"]
    expected_files = {".lock", "study.json", "selected-settings.json"} | {"source/"+name for name in source_hashes}
    if set(manifest["files"]) != expected_files:
        raise ValueError("Changed comparison source inventory")
    for name, sha in source_hashes.items():
        if manifest["files"]["source/"+name] != sha:
            raise ValueError("Changed retained source binding")
    if any(source_hashes.get(name) != sha for name, sha in SELECTED_SOURCE_SHA256.items()):
        raise ValueError("Selected scientific source differs")
    inventory, selected = expected["inventory"], _selected(selected_bytes)
    if set(manifest["children"]) != {_base_name(base) for base in _groups(inventory)}:
        raise ValueError("Changed base inventory")
    if _prefix(root, inventory, selected, _development_material) != len(case_inventory(inventory)):
        raise ValueError("Incomplete global comparison inventory")
    outcomes = []
    for case in case_inventory(inventory):
        folder = root/_base_name(case["base_wavefront_id"])/case["id"]
        checked, sha = _verify_case(folder, case, selected, _development_material(case))
        outcomes.append({"case": case, "manifest_sha256": sha, **checked})
    return {"schema": 1, "mode": DEVELOPMENT, "inventory": inventory,
            "raw_manifest_sha256": _sha(pilot._read_selected(root/"manifest.json")),
            "study": study, "outcomes": outcomes,
            "artifact_verification": "complete_native_consistency_checks",
            "physical_recomputation": False, "held_out_provenance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--selected-settings", type=Path)
    parser.add_argument("--operating-reserve-bytes", type=int)
    parser.add_argument("--limit-cases", type=int, default=36)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        if args.selected_settings is not None or args.operating_reserve_bytes is not None or args.resume:
            parser.error("Verification accepts only an existing output directory")
        verify_development(args.output)
    else:
        if args.selected_settings is None or args.operating_reserve_bytes is None:
            parser.error("Development acquisition requires selected settings and an explicit reserve")
        run_development(args.output, pilot._read_selected(args.selected_settings),
                        operating_reserve_bytes=args.operating_reserve_bytes,
                        resume=args.resume, limit_cases=args.limit_cases)


if __name__ == "__main__":
    main()
