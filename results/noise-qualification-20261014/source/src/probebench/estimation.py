"""Noise-aware phase diversity using public optics and camera calibration only."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .likelihood import MEAN_FLOOR, TAIL_RELATIVE_TOLERANCE, poisson_gaussian_nll


def phase_objective(model, observations, read_noise_electrons, scaled_coefficients):
    """NLL per expected photon, gradient with respect to 100-nm coefficients."""
    estimate = 100 * np.asarray(scaled_coefficients, dtype=np.float64)
    objective, gradient = 0.0, np.zeros(6)
    photons = sum(o.expected_photons for o in observations)
    if photons <= 0:
        raise ValueError("Positive total exposure photons required")
    for observation in observations:
        intensity, jacobian = model.intensity_and_jacobian(estimate - observation.applied_opd_nm)
        nll, pixel_gradient = poisson_gaussian_nll(
            observation.expected_photons * intensity,
            observation.image_electrons, read_noise_electrons)
        objective += nll
        gradient += 100 * observation.expected_photons * np.einsum("ij,mij->m", pixel_gradient, jacobian)
    return objective / photons, gradient / photons


def estimate_phase_mle(model, observations, read_noise_electrons, coefficient_limit_nm=200.0,
                       max_iterations=200):
    if len(observations) < 2:
        raise ValueError("At least two known diverse measurements are required")
    commands = np.asarray([o.applied_opd_nm for o in observations])
    if np.all(commands == commands[0]):
        raise ValueError("Known diverse commands are required")
    if not np.isfinite(coefficient_limit_nm) or coefficient_limit_nm <= 0:
        raise ValueError("Coefficient limit must be finite and positive")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    start_offset = min(0.5, coefficient_limit_nm / 200)
    starts = [np.zeros(6), np.array([start_offset, 0, 0, 0, 0, 0]),
              np.array([-start_offset, 0, 0, 0, 0, 0])]
    fits = [minimize(lambda x: phase_objective(model, observations, read_noise_electrons, x),
                    start, method="L-BFGS-B", jac=True,
                    bounds=[(-coefficient_limit_nm/100, coefficient_limit_nm/100)] * 6,
                    options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-8, "maxls": 30})
            for start in starts]
    finite = [i for i, fit in enumerate(fits) if np.isfinite(fit.fun) and np.isfinite(fit.x).all()]
    if not finite:
        raise FloatingPointError("No finite phase-diversity fit; retain the episode as a failure")
    chosen = min(finite, key=lambda i: fits[i].fun)
    best = fits[chosen]
    return 100 * best.x, {
        "objective": "poisson_gaussian_convolution_nll_per_expected_photon",
        "cost": float(best.fun), "optimizer_success": bool(best.success),
        "chosen_start": chosen, "mean_floor_counts": MEAN_FLOOR,
        "relative_likelihood_tail_tolerance": TAIL_RELATIVE_TOLERANCE,
        "reported_function_evaluations": sum(int(f.nfev) for f in fits),
        "starts": [{"initial_opd_nm": (100*start).tolist(), "success": bool(fit.success),
                    "status": int(fit.status), "message": str(fit.message),
                    "iterations": int(fit.nit), "cost": float(fit.fun) if np.isfinite(fit.fun) else None,
                    "finite": bool(np.isfinite(fit.fun) and np.isfinite(fit.x).all())}
                   for start, fit in zip(starts, fits)],
    }
