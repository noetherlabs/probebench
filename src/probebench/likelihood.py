"""Poisson plus Gaussian read-noise likelihood with controlled sum truncation.

Read docs/noise-likelihood.md for the likelihood, tail bound and numerical floor.
The truncation bound excludes floating-point roundoff and the explicit mean floor.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

MEAN_FLOOR = 1e-12
TAIL_RELATIVE_TOLERANCE = 1e-12


def tail_halfwidth(read_noise, tolerance=TAIL_RELATIVE_TOLERANCE):
    if not np.isfinite(read_noise) or read_noise <= 0:
        raise ValueError("Positive finite read noise is required for convolution")
    if not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Tail tolerance must lie strictly between zero and one")
    variance = read_noise**2
    halfwidth = max(1, int(np.ceil(read_noise * np.sqrt(2 * np.log(2 / tolerance)))))
    while True:
        bound = (2 * np.exp(-halfwidth * (halfwidth + 1) / (2 * variance))
                 / -np.expm1(-(halfwidth + 1) / variance))
        if bound <= tolerance:
            return halfwidth, float(bound)
        halfwidth += 1


def _posterior_mode(mean, observed, variance):
    """First integer k for which the adjacent posterior mass ratio is <= 1."""
    lower = np.zeros(mean.shape, dtype=np.int64)
    upper = np.maximum(0, np.ceil(observed + variance * np.log1p(mean))).astype(np.int64) + 1
    log_mean = np.log(mean)
    while np.any(lower < upper):
        middle = lower + (upper - lower) // 2
        increasing = log_mean - np.log(middle + 1) + (observed - middle - 0.5) / variance > 0
        lower = np.where(increasing & (lower < upper), middle + 1, lower)
        upper = np.where(~increasing & (lower < upper), middle, upper)
    return lower


def poisson_gaussian_nll(mean_counts, image_electrons, read_noise):
    """Return total negative log likelihood and d(total NLL)/d(mean counts).

    Values below MEAN_FLOOR use a constant floor and zero derivative. This explicit
    numerical regularization avoids singular derivatives at exact dark pixels.
    Inputs have equal shapes; negative camera samples remain valid with read noise.
    """
    original = np.asarray(mean_counts, dtype=np.float64)
    observed = np.asarray(image_electrons, dtype=np.float64)
    if original.shape != observed.shape or original.size == 0:
        raise ValueError("Mean and image must have the same nonempty shape")
    if not np.isfinite(original).all() or np.any(original < 0) or not np.isfinite(observed).all():
        raise ValueError("Finite data and nonnegative mean counts are required")
    if not np.isfinite(read_noise) or read_noise < 0:
        raise ValueError("Read noise must be finite and nonnegative")
    mean = np.maximum(original, MEAN_FLOOR)
    if read_noise == 0:
        if np.any(observed < 0) or np.any(observed != np.floor(observed)):
            raise ValueError("A pure Poisson observation must contain nonnegative integer counts")
        value = np.sum(mean - observed * np.log(mean) + gammaln(observed + 1))
        gradient = 1 - observed / mean
    else:
        variance = read_noise**2
        shape = mean.shape
        mu, y = mean.ravel(), observed.ravel()
        mode = _posterior_mode(mu, y, variance)
        halfwidth, _ = tail_halfwidth(read_noise)
        counts = mode[:, None] + np.arange(-halfwidth, halfwidth + 1)
        valid = counts >= 0
        k = np.maximum(counts, 0)
        log_weights = (k * np.log(mu[:, None]) - mu[:, None] - gammaln(k + 1)
                       - (y[:, None] - k)**2 / (2 * variance)
                       - 0.5 * np.log(2 * np.pi * variance))
        log_weights = np.where(valid, log_weights, -np.inf)
        log_density = logsumexp(log_weights, axis=1)
        expected_count = np.sum(k * np.exp(log_weights - log_density[:, None]), axis=1)
        value = -log_density.sum()
        gradient = (1 - expected_count / mu).reshape(shape)
    return float(value), np.where(original < MEAN_FLOOR, 0.0, gradient)
