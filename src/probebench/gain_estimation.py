"""Joint phase/command-response estimation; no evaluator truth is accepted.

The gain here multiplies commanded OPD. It is NOT pyramid-WFS optical gain.
This is an offline calibration prototype, not a qualified runtime controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence
import numpy as np
from scipy.optimize import minimize
from .likelihood import poisson_gaussian_nll
from .optics import coefficients, OpticalModel


@dataclass(frozen=True)
class NominalObservation:
    """Camera data and nominal commands only; never physical response labels."""
    nominal_command_nm: np.ndarray
    image_electrons: np.ndarray
    expected_photons: float

    def __post_init__(self):
        command = coefficients(self.nominal_command_nm)
        image = np.array(self.image_electrons, dtype=float, copy=True)
        if image.ndim != 2 or not image.size or not np.isfinite(image).all():
            raise ValueError('A finite two-dimensional camera array is required')
        if not np.isfinite(self.expected_photons) or self.expected_photons <= 0:
            raise ValueError('Expected photons must be positive and finite')
        command.setflags(write=False)
        image.setflags(write=False)
        object.__setattr__(self, 'nominal_command_nm', command)
        object.__setattr__(self, 'image_electrons', image)

    @property
    def applied_opd_nm(self):
        """Legacy estimator adapter: nominal value, not actual physical OPD.

        The legacy estimator assumes calibration is correct. This property exists
        only to compare it on the same camera data without leaking true gain.
        """
        return self.nominal_command_nm


def joint_objective(model: OpticalModel, observations: Sequence[NominalObservation],
                    read_noise_electrons: float, parameters: np.ndarray):
    """Exact Poisson-Gaussian NLL; x[:6]=phase/100, x[6]=(gain-1)/0.1."""
    x = np.asarray(parameters, dtype=float)
    if x.shape != (7,) or not np.isfinite(x).all():
        raise ValueError('Seven finite scaled parameters are required')
    phase, gain = 100*x[:6], 1 + 0.1*x[6]
    photons = sum(o.expected_photons for o in observations)
    if photons <= 0 or gain <= 0:
        raise ValueError('Positive photons and gain are required')
    cost, gradient = 0.0, np.zeros(7)
    for obs in observations:
        intensity, jac = model.intensity_and_jacobian(phase-gain*obs.nominal_command_nm)
        value, pixel_gradient = poisson_gaussian_nll(
            obs.expected_photons*intensity, obs.image_electrons, read_noise_electrons)
        derivative = obs.expected_photons*np.einsum('ij,mij->m', pixel_gradient, jac)
        cost += value
        gradient[:6] += 100*derivative
        gradient[6] -= 0.1*np.dot(obs.nominal_command_nm, derivative)
    return cost/photons, gradient/photons


def estimate_phase_and_gain(model: OpticalModel,
                            observations: Sequence[NominalObservation],
                            read_noise_electrons: float,
                            coefficient_limit_nm: float = 200.0,
                            gain_bounds: tuple[float, float] = (0.7, 1.3),
                            max_iterations: int = 200):
    """Fit six phase coefficients and one positive scalar response from images.

    A bounded gain prior and command diversity are required, but do not guarantee
    global identifiability. All local-fit outcomes are returned. The finite best
    fit is retained even when its optimizer reports failure, matching the existing
    nominal estimator; callers must check optimizer_success before deployment.
    """
    if len(observations) < 2:
        raise ValueError('At least two observations are required')
    if (not np.isfinite(coefficient_limit_nm) or coefficient_limit_nm <= 0
            or type(max_iterations) is not int or max_iterations <= 0):
        raise ValueError('Positive coefficient and iteration limits are required')
    lo, hi = gain_bounds
    if not (np.isfinite([lo, hi]).all() and 0 < lo < 1 < hi):
        raise ValueError('Gain bounds must be finite, positive, and contain one')
    commands = np.array([o.nominal_command_nm for o in observations])
    if np.all(commands == commands[0]):
        raise ValueError('Unchanging commands cannot separate phase from command gain')
    expected_shape = (model.config.pupil_pixels*model.config.oversampling,)*2
    if any(o.image_electrons.shape != expected_shape for o in observations):
        raise ValueError('Camera shape does not match the optical model')
    if not np.isfinite(read_noise_electrons) or read_noise_electrons < 0:
        raise ValueError('Read noise must be finite and nonnegative')
    bounds = [(-coefficient_limit_nm/100, coefficient_limit_nm/100)]*6
    bounds += [((lo-1)/0.1, (hi-1)/0.1)]
    offset = min(0.5, coefficient_limit_nm/200)
    fits, records = [], []
    started = perf_counter()
    for sign in (0, 1, -1):
        initial = np.zeros(7)
        initial[0] = sign*offset
        fit = minimize(lambda x: joint_objective(model, observations, read_noise_electrons, x),
                       initial, jac=True, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iterations, 'ftol': 1e-12,
                                'gtol': 1e-8, 'maxls': 30})
        finite = bool(np.isfinite(fit.fun) and np.isfinite(fit.x).all())
        fits.append(fit)
        records.append({'initial_opd_nm':(100*initial[:6]).tolist(),
                        'initial_gain':1.0, 'success':bool(fit.success),
                        'finite':finite, 'cost':float(fit.fun) if finite else None,
                        'parameters':fit.x.tolist() if finite else None,
                        'function_evaluations':int(fit.nfev),
                        'iterations':int(fit.nit), 'message':str(fit.message)})
    valid = [i for i, r in enumerate(records) if r['finite']]
    if not valid:
        raise FloatingPointError('No finite joint fit; retain this episode as a failure')
    chosen = min(valid, key=lambda i:fits[i].fun)
    best = fits[chosen]
    gain = float(1+0.1*best.x[6])
    diagnostics = {'optimizer_success':bool(best.success), 'chosen_start':chosen,
                   'cost':float(best.fun), 'starts':records,
                   'reported_function_evaluations':sum(r['function_evaluations'] for r in records),
                   'fit_elapsed_seconds':perf_counter()-started,
                   'gain_at_bound':bool(min(gain-lo, hi-gain) < 1e-5),
                   'gain_bounds':list(gain_bounds),
                   'objective':'poisson_gaussian_convolution_nll_per_expected_photon',
                   'qualification':'development_only_not_global_identifiability_or_runtime_assurance'}
    return 100*best.x[:6], gain, diagnostics


def compensated_command(phase_nm: np.ndarray, gain: float, limit_nm: float = 200.0):
    """Convert estimated physical phase to nominal actuator command, with bounds."""
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(limit_nm) or limit_nm <= 0:
        raise ValueError('Gain and command limit must be positive and finite')
    return np.clip(coefficients(phase_nm)/gain, -limit_nm, limit_nm)
