"""Development baselines. Only public calibration and observations enter policies."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def visible_score(observation):
    """Fixed central 3x3 flux fraction; uses no true wavefront or noiseless field."""
    n = observation.image_electrons.shape[0]
    return float(observation.image_electrons[n//2-1:n//2+2, n//2-1:n//2+2].sum()
                 / observation.expected_photons)


def estimate_phase(model, observations, coefficient_limit_nm=200.0):
    """Multistart calibrated-intensity least squares, in units of 100 nm.

    This initial baseline is not a Poisson-Gaussian maximum likelihood estimator.
    Its convergence and noisy efficiency require qualification before final cohorts.
    """
    if not np.isfinite(coefficient_limit_nm) or coefficient_limit_nm <= 0:
        raise ValueError("Coefficient limit must be finite and positive")
    if len(observations) < 2:
        raise ValueError("Phase diversity requires at least two measurements")
    commands = np.array([o.applied_opd_nm for o in observations])
    if np.all(commands == commands[0]):
        raise ValueError("Known diverse commands are required")

    def residual(scaled):
        estimate = 100 * scaled
        return np.concatenate([
            (model.intensity(estimate - o.applied_opd_nm)
             - o.image_electrons / o.expected_photons).ravel()
            for o in observations
        ])

    start_offset = min(0.5, coefficient_limit_nm / 200)
    starts = [np.zeros(6), np.array([start_offset, 0, 0, 0, 0, 0]),
              np.array([-start_offset, 0, 0, 0, 0, 0])]
    fits = [least_squares(residual, x, bounds=(-coefficient_limit_nm/100,
            coefficient_limit_nm/100), max_nfev=160, ftol=1e-10, xtol=1e-10,
            gtol=1e-10) for x in starts]
    best = min(fits, key=lambda f: f.cost)
    return 100 * best.x, {
        "objective": "unweighted_calibrated_intensity_least_squares",
        "cost": float(best.cost),
        "optimizer_success": bool(best.success),
        "optimizer_status": int(best.status),
        "reported_function_evaluations": sum(int(f.nfev) for f in fits),
        "starts": len(starts),
    }


class Policy:
    """Simple policy interface shared by all four development references."""

    def __init__(self, name, model, config, seed):
        if name not in ("no_action", "random_probe", "coordinate_search", "phase_diversity"):
            raise ValueError(f"Unknown policy: {name}")
        self.name, self.model, self.config = name, model, config
        self.rng = np.random.default_rng(seed)
        self.fit_records = []

    def command(self, history):
        if not history or self.name == "no_action":
            return np.zeros(6)
        if self.name == "random_probe":
            limit = min(60, self.config.command_limit_nm)
            return self.rng.uniform(-limit, limit, size=6)
        if self.name == "coordinate_search":
            best = max(history, key=visible_score)
            command = np.array(best.applied_opd_nm)
            step = len(history) - 1
            command[(step // 2) % 6] += 40 * (1 if step % 2 == 0 else -1)
            return np.clip(command, -self.config.command_limit_nm, self.config.command_limit_nm)
        if len(history) == 1:
            return np.array([min(80., self.config.command_limit_nm), 0, 0, 0, 0, 0])
        estimate, diagnostics = estimate_phase(self.model, history, self.config.command_limit_nm)
        self.fit_records.append(diagnostics)
        return estimate

    def final_command(self, history):
        if self.name == "no_action":
            return np.zeros(6)
        if self.name == "phase_diversity":
            # Submit the last measured correction; do not gain an unmeasured final move.
            return np.array(history[-1].applied_opd_nm)
        return np.array(max(history, key=visible_score).applied_opd_nm)
