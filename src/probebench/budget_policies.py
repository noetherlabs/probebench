"""Classical development policies with complete six-mode and budget semantics."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
from time import perf_counter

import numpy as np

from .estimation import estimate_phase_mle
from .optics import coefficients
from .policies import visible_score

METHODS = ("no_action_v2", "random_probe_v2", "coordinate_search_v2", "phase_diversity_mle_v1")


@dataclass(frozen=True)
class PolicyConfig:
    random_amplitude_nm: float = 60.0
    coordinate_step_nm: float = 40.0
    coordinate_step_floor_nm: float = 5.0
    diversity_probe_nm: float = 80.0
    max_mle_iterations: int = 200

    def __post_init__(self):
        for name in ("random_amplitude_nm", "coordinate_step_nm", "coordinate_step_floor_nm",
                     "diversity_probe_nm"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.coordinate_step_floor_nm > self.coordinate_step_nm:
            raise ValueError("Coordinate step floor exceeds initial step")
        if type(self.max_mle_iterations) is not int or self.max_mle_iterations < 1:
            raise ValueError("Positive integer MLE iteration budget required")


def visible_history_digest(history):
    """Cache identity contains public observations only; a changed frame forces a refit."""
    digest = hashlib.sha256()
    for observation in history:
        digest.update(json.dumps(observation.metadata(), sort_keys=True, allow_nan=False).encode())
        image = np.asarray(observation.image_electrons, dtype="<f8")
        digest.update(json.dumps(image.shape).encode())
        digest.update(image.tobytes())
    return digest.hexdigest()


class ClassicalPolicy:
    """Accept only public optics, instrument limits, camera data and policy randomness."""

    def __init__(self, name, model, episode_config, policy_config=None, policy_seed=0):
        if name not in METHODS:
            raise ValueError(f"Unknown classical policy: {name}")
        if type(policy_seed) is not int or policy_seed < 0:
            raise ValueError("Policy seed must be a nonnegative integer")
        self.name, self.model, self.instrument = name, model, episode_config
        self.config = policy_config or PolicyConfig()
        self.seed = policy_seed
        self.fit_records = []
        self._estimates = {}

    def restore_fit_records(self, records, history):
        """Restore checked local evidence; never use this with untrusted agent outputs."""
        restored = {}
        for record in records:
            count = record["observations_used"]
            if type(count) is not int or not 2 <= count <= len(history):
                raise ValueError("Invalid saved estimator observation prefix")
            key = visible_history_digest(history[:count])
            estimate = coefficients(record["estimated_opd_nm"])
            if (key != record["visible_history_sha256"] or key in restored
                or record["fit_status"] not in {"finite", "numerical_failure"}
                or np.max(np.abs(estimate)) > self.instrument.command_limit_nm):
                raise ValueError("Saved estimator record does not match public history")
            restored[key] = estimate.copy()
        self.fit_records = deepcopy(records)
        self._estimates = restored

    def _bound(self, command):
        return np.clip(command, -self.instrument.command_limit_nm, self.instrument.command_limit_nm)

    def _estimate(self, history):
        if len(history) < 2:
            return np.zeros(6)
        key = visible_history_digest(history)
        if key not in self._estimates:
            started = perf_counter()
            try:
                estimate, diagnostics = estimate_phase_mle(self.model, history,
                    self.instrument.read_noise_electrons, self.instrument.command_limit_nm,
                    max_iterations=self.config.max_mle_iterations)
                record = {"fit_status": "finite", **diagnostics}
            except FloatingPointError as error:
                # Retain failure and its camera-only fallback; never silently drop the episode.
                estimate = np.array(max(history, key=visible_score).applied_opd_nm)
                record = {"fit_status": "numerical_failure", "reason": str(error),
                          "fallback": "best_measured_camera_score", "optimizer_success": False,
                          **getattr(error, "diagnostics", {})}
            self.fit_records.append({"observations_used": len(history),
                "visible_history_sha256": key, "fit_elapsed_seconds": perf_counter()-started,
                "estimated_opd_nm": estimate.tolist(), **record})
            self._estimates[key] = estimate.copy()
        return self._estimates[key].copy()

    def command(self, history):
        count = len(history)
        if not count or self.name == "no_action_v2":
            return np.zeros(6)
        if self.name == "random_probe_v2":
            # Public seed and observation index make diagnostic calls order-independent.
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, count]))
            return self._bound(rng.uniform(-self.config.random_amplitude_nm,
                                            self.config.random_amplitude_nm, 6))
        if self.name == "coordinate_search_v2":
            incumbent = history[0]
            for first in range(1, count-1, 2):
                incumbent = max((incumbent, history[first], history[first+1]), key=visible_score)
            pair = (count-1)//2
            mode, sweep = pair % 6, pair // 6
            step = max(self.config.coordinate_step_floor_nm,
                       self.config.coordinate_step_nm * 2.0**(-sweep))
            command = np.array(incumbent.applied_opd_nm, dtype=float)
            command[mode] += step * (1 if (count-1) % 2 == 0 else -1)
            return self._bound(command)
        if count == 1:
            return self._bound(np.array([self.config.diversity_probe_nm, 0, 0, 0, 0, 0]))
        command = self._estimate(history)
        if count % 2 == 1:
            # First post-correction probe is negative; subsequent probes alternate sign.
            sign = -1 if ((count-3)//2) % 2 == 0 else 1
            command[0] += sign * self.config.diversity_probe_nm
        return self._bound(command)

    def final_command(self, history):
        if not history:
            raise ValueError("Final submission requires at least one measurement")
        if self.name == "no_action_v2":
            return np.zeros(6)
        if self.name == "phase_diversity_mle_v1":
            return self._estimate(history)
        return np.array(max(history, key=visible_score).applied_opd_nm)
