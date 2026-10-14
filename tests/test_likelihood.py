import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm, poisson

from probebench.likelihood import MEAN_FLOOR, poisson_gaussian_nll, tail_halfwidth


@pytest.mark.parametrize("noise", [0.25, 1.0, 3.0])
def test_convolution_matches_independent_wide_distribution_sum(noise):
    means = np.array([1e-8, .1, 2., 50., 1000., 1e-3, 10_000.])
    measured = np.array([-3., .3, 1.7, 55., 980., 95., -2.])
    expected = 0.0
    for mu, y in zip(means, measured):
        last = int(max(mu + 15*np.sqrt(mu+1), y + 20*noise, 100))
        counts = np.arange(last + 1)
        expected -= logsumexp(poisson.logpmf(counts, mu) + norm.logpdf(y-counts, scale=noise))
    actual, _ = poisson_gaussian_nll(means, measured, noise)
    assert actual == pytest.approx(expected, rel=2e-12, abs=2e-9)
    assert tail_halfwidth(noise)[1] <= 1e-12


@pytest.mark.parametrize("mu,y,noise", [(.01, -1.2, 1), (1.5, .7, 1),
                                       (100., 101.2, 1), (2., 4.3, 3)])
def test_pixel_likelihood_derivative_matches_finite_difference(mu, y, noise):
    step = max(1e-6, mu*1e-5)
    plus = poisson_gaussian_nll(np.array([mu+step]), np.array([y]), noise)[0]
    minus = poisson_gaussian_nll(np.array([mu-step]), np.array([y]), noise)[0]
    gradient = poisson_gaussian_nll(np.array([mu]), np.array([y]), noise)[1][0]
    assert gradient == pytest.approx((plus-minus)/(2*step), rel=2e-5, abs=2e-7)


def test_pure_poisson_and_explicit_dark_pixel_floor():
    means, measured = np.array([.1, 1, 100]), np.array([0., 3., 95.])
    value, gradient = poisson_gaussian_nll(means, measured, 0)
    assert value == pytest.approx(-poisson.logpmf(measured, means).sum())
    np.testing.assert_allclose(gradient, 1-measured/means)
    value, gradient = poisson_gaussian_nll(np.array([0., MEAN_FLOOR/2]), np.array([-1., 2.]), 1)
    assert value == pytest.approx(-norm.logpdf([-1., 2.]).sum(), abs=1e-10)
    np.testing.assert_array_equal(gradient, [0., 0.])


@pytest.mark.parametrize("mean,image,noise", [([-1], [0], 1), ([1], [np.nan], 1),
    ([1,2], [1], 1), ([1], [-1], 0), ([1], [.1], 0), ([1], [1], -1)])
def test_invalid_noise_inputs_rejected(mean, image, noise):
    with pytest.raises(ValueError): poisson_gaussian_nll(mean, image, noise)
