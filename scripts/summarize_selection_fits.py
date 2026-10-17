"""Summarize retained optimizer diagnostics from classical-selection-v1-20261016.

This descriptive check binds case JSON to the published native manifest. It does
not execute archived code, verify camera pixels, or rerun optical optimization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat

MANIFEST_SHA256 = "95130d4d9bf5d534aa3bfc88bdaef5934b5862769061df2bb26c03cf72806ef0"
PHOTONS = (2000, 20000)
BUDGETS = (3, 7, 13)
BASES = tuple((stratum, f"rms_selection_v1:{stratum}:{i:06d}")
              for stratum in ("low", "medium", "high") for i in range(4))
CANDIDATES = {
    "no_action_default": "no_action_v2", "unprobed_default": "unprobed_mle_v1",
    **{f"random_amp_{n:03d}": "random_probe_v2" for n in (20, 40, 80)},
    **{f"coordinate_step_{n:03d}": "coordinate_search_v2" for n in (20, 40, 80)},
    **{f"fixed_probe_{n:03d}": "fixed_diversity_mle_v1" for n in (40, 80, 120)},
    **{f"adaptive_probe_{n:03d}": "phase_diversity_mle_v1" for n in (40, 80, 120)},
}
COUNT_FIELDS = (
    "episodes", "exposures", "multistart_fit_calls", "attempted_starts",
    "finite_successful_starts", "finite_unconverged_starts", "nonfinite_starts",
    "finite_converged_fit_calls", "finite_unconverged_fit_calls", "numerical_failure_fit_calls",
    "fit_calls_with_unsuccessful_starts", "episodes_with_unsuccessful_starts",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute_path(value):
    path = Path(value)
    require(".." not in path.parts, "Parent traversal is not permitted")
    return Path(os.path.abspath(path))


@contextmanager
def directory(path):
    """Open each directory component without following symlinks."""
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def read_file(parent_fd, name):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent_fd)
    with os.fdopen(fd, "rb") as handle:
        require(stat.S_ISREG(os.fstat(handle.fileno()).st_mode), "Expected a regular file")
        data = handle.read(16 * 1024 * 1024 + 1)
        require(len(data) <= 16 * 1024 * 1024, "JSON file exceeds size bound")
        return data


def decode(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant")

    return json.loads(data, object_pairs_hook=unique, parse_constant=reject_constant)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def count_fit(fit, counts, statuses, issues, identity, fit_index):
    starts = fit["starts"]
    require(type(starts) is list and len(starts) == 3, "Each fit requires all three starts")
    require(type(fit["optimizer_success"]) is bool, "Malformed chosen-start success")
    require(fit["fit_status"] in ("finite", "numerical_failure"), "Unknown fit status")
    usable, unsuccessful = [], False
    for index, start in enumerate(starts):
        require(type(start["finite"]) is bool and type(start["success"]) is bool,
                "Malformed start outcome")
        require(type(start["status"]) is int or start["status"] is None, "Malformed solver status")
        require(type(start["message"]) is str, "Malformed stopping reason")
        solution = start["solution_opd_nm"]
        require(start["cost"] is None or finite(start["cost"]), "Malformed start objective")
        require(solution is None or (type(solution) is list and len(solution) == 6
                and all(value is None or finite(value) for value in solution)), "Malformed start solution")
        actual_finite = finite(start["cost"]) and type(solution) is list and all(finite(value) for value in solution)
        require(start["finite"] == actual_finite, "Inconsistent start finiteness")
        if start["finite"]:
            usable.append(index)
        category = ("finite_successful_starts" if start["finite"] and start["success"] else
                    "finite_unconverged_starts" if start["finite"] else "nonfinite_starts")
        counts["attempted_starts"] += 1
        counts[category] += 1
        statuses[(category, start["success"], start["status"], start["message"])] += 1
        if not start["finite"] or not start["success"]:
            unsuccessful = True
            issues.append({**identity, "fit_index": fit_index, "start_index": index,
                           "category": category, "status": start["status"],
                           "reported_success": start["success"],
                           "stopping_reason": start["message"],
                           "chosen": index == fit["chosen_start"]})
    counts["multistart_fit_calls"] += 1
    if usable:
        chosen = min(usable, key=lambda index: starts[index]["cost"])
        require(type(fit["chosen_start"]) is int and fit["chosen_start"] == chosen
                and fit["fit_status"] == "finite" and fit["cost"] == starts[chosen]["cost"]
                and fit["optimizer_success"] == starts[chosen]["success"],
                "Chosen start disagrees with minimum finite objective")
        counts["finite_converged_fit_calls" if fit["optimizer_success"] else
               "finite_unconverged_fit_calls"] += 1
    else:
        require(fit["chosen_start"] is None and fit["cost"] is None
                and fit["fit_status"] == "numerical_failure" and not fit["optimizer_success"],
                "Failed fit has inconsistent chosen-start accounting")
        counts["numerical_failure_fit_calls"] += 1
    counts["fit_calls_with_unsuccessful_starts"] += unsuccessful
    return unsuccessful


def status_rows(statuses):
    return [{"category": key[0], "reported_success": key[1], "status": key[2],
             "stopping_reason": key[3], "count": count}
            for key, count in sorted(statuses.items(), key=lambda item: str(item[0]))]


def summarize(archive, output):
    archive, output = absolute_path(archive), absolute_path(output)
    require(output != archive and archive not in output.parents and output not in archive.parents,
            "Output must be outside the archive")
    # The destination parent must already exist; O_EXCL below prevents replacement.
    with directory(output.parent) as destination_fd, directory(archive) as archive_fd:
        require(output.name not in os.listdir(destination_fd), "Output already exists")
        lock_fd = os.open(".lock", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=archive_fd)
        with os.fdopen(lock_fd, "rb") as lock:
            require(stat.S_ISREG(os.fstat(lock.fileno()).st_mode), "Expected a regular archive lock")
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            manifest_bytes = read_file(archive_fd, "manifest.json")
            require(sha256(manifest_bytes) == MANIFEST_SHA256, "Unexpected native manifest hash")
            manifest = decode(manifest_bytes)
            files = manifest["files"]
            require(type(files) is dict, "Malformed file inventory")
            for name, digest in files.items():
                relative = PurePosixPath(name)
                require(not relative.is_absolute() and str(relative) == name
                        and ".." not in relative.parts and "\\" not in name
                        and re.fullmatch(r"[0-9a-f]{64}", digest), "Malformed manifest path or hash")
            observed = set()
            for base, directories, names, dir_fd in os.fwalk(".", dir_fd=archive_fd, follow_symlinks=False):
                for name in directories + names:
                    mode = os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode
                    require(not stat.S_ISLNK(mode), "Archive symlinks are not permitted")
                    require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), "Unexpected archive file type")
                for name in names:
                    observed.add((PurePosixPath(base) / name).as_posix().removeprefix("./"))
            require(observed == set(files) | {"manifest.json"}, "Archive file inventory mismatch")
            expected = set((photons, budget, stratum, base, member, replicate)
                           for photons, budget, (stratum, base), member, replicate in
                           itertools.product(PHOTONS, BUDGETS, BASES, ("primary", "conjugate"), (0, 1)))
            expected_count = len(expected) * len(CANDIDATES)
            case_paths = sorted(name for name in files if name.startswith("cases/"))
            require(len(case_paths) == manifest["episode_count"] == expected_count,
                    "Incomplete case inventory")
            counts = {candidate: Counter({key: 0 for key in COUNT_FIELDS}) for candidate in CANDIDATES}
            statuses = defaultdict(Counter)
            coverage = defaultdict(set)
            issues = []
            case_fd = os.open("cases", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=archive_fd)
            try:
                for name in case_paths:
                    require(re.fullmatch(r"cases/case_[0-9a-f]{64}\.json", name), "Malformed case path")
                    data = read_file(case_fd, PurePosixPath(name).name)
                    require(sha256(data) == files[name], "Case hash mismatch")
                    record = decode(data)
                    case, result = record["case"], record["result"]
                    candidate = case["candidate_id"]
                    require(candidate in CANDIDATES and case["id"] == PurePosixPath(name).stem
                            and case["family"] == case["method"] == CANDIDATES[candidate],
                            "Case identity mismatch")
                    for key in ("photons_per_exposure", "exposure_budget", "replicate"):
                        require(type(case[key]) is int, "Malformed case grid value")
                    semantic = tuple(case[key] for key in ("photons_per_exposure", "exposure_budget",
                                     "stratum_id", "base_wavefront_id", "member", "replicate"))
                    require(semantic in expected and semantic not in coverage[candidate],
                            "Duplicate or unexpected semantic case")
                    coverage[candidate].add(semantic)
                    budget = case["exposure_budget"]
                    require(type(result["exposures"]) is int and result["exposures"] == budget,
                            "Exposure accounting mismatch")
                    fits = result["fit_records"]
                    family = CANDIDATES[candidate]
                    prefixes = (list(range(2, budget + 1)) if family == "phase_diversity_mle_v1" else
                                [budget] if family in ("fixed_diversity_mle_v1", "unprobed_mle_v1") else [])
                    require(type(fits) is list and len(fits) == len(prefixes), "Incomplete fit schedule")
                    candidate_counts = counts[candidate]
                    candidate_counts.update(episodes=1, exposures=budget)
                    affected = False
                    for fit_index, (fit, prefix) in enumerate(zip(fits, prefixes)):
                        require(type(fit["observations_used"]) is int and fit["observations_used"] == prefix,
                                "Fit prefix mismatch")
                        affected |= count_fit(fit, candidate_counts, statuses[candidate], issues,
                                              {"candidate_id": candidate, "case_id": case["id"]}, fit_index)
                    candidate_counts["episodes_with_unsuccessful_starts"] += affected
            finally:
                os.close(case_fd)
            require(all(coverage[candidate] == expected for candidate in CANDIDATES), "Incomplete case grid")
            total = {key: sum(counts[candidate][key] for candidate in CANDIDATES) for key in COUNT_FIELDS}
            require(total["exposures"] == manifest["exposure_count"], "Total exposure mismatch")
            total_statuses = sum(statuses.values(), Counter())
            result = {
                "schema": "probebench_selection_fit_diagnostics_v1",
                "native_manifest_sha256": MANIFEST_SHA256,
                "independent_base_wavefronts": len(BASES),
                "scope": "Manifest-bound descriptive optimizer diagnostics; no optical replay or optimizer rerun.",
                "total": {**total, "start_statuses": status_rows(total_statuses)},
                "candidates": [{"candidate_id": candidate, "family": CANDIDATES[candidate],
                                **counts[candidate], "start_statuses": status_rows(statuses[candidate])}
                               for candidate in sorted(CANDIDATES)],
                "unsuccessful_starts": issues,
                "interpretation": [
                    "Fit calls and individual optimizer starts have different denominators.",
                    "Counts include all parameter candidates and every unchosen start.",
                    "Optimizer success records a stopping condition, not global optimality or identifiability.",
                    "Episodes reuse base wavefronts and are not independent wavefront samples."
                ],
            }
            payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            require(read_file(archive_fd, "manifest.json") == manifest_bytes, "Manifest changed during analysis")
            fd = os.open(output.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o644, dir_fd=destination_fd)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
            except BaseException:
                os.unlink(output.name, dir_fd=destination_fd)
                raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = summarize(args.archive, args.output)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Diagnostic summary rejected: {type(error).__name__}: {error}\n")
    print(f"Summarized {result['total']['multistart_fit_calls']} fit calls and "
          f"{result['total']['attempted_starts']} starts.")


if __name__ == "__main__":
    main()
