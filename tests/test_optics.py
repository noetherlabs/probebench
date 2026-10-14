import numpy as np
import pytest

from probebench.optics import OpticalModel, OpticsConfig, coefficients, opd_to_surface_nm


@pytest.fixture
def model():
    return OpticalModel(OpticsConfig())


def test_unit_power_piston_free_modes_and_actual_pupil_rms(model):
    np.testing.assert_allclose(model.basis[:, model.pupil].mean(axis=1), 0, atol=1e-15)
    np.testing.assert_allclose(np.mean(model.basis[:, model.pupil]**2, axis=1), 1, atol=1e-14)
    wave = np.array([35, 10, -12, 8, 6, -15])
    assert model.intensity(wave).sum() == pytest.approx(1, abs=2e-15)
    assert model.evaluate(np.zeros(6))["strehl_peak_ratio"] == pytest.approx(1)
    assert model.evaluate(wave)["residual_wfe_nm"] == pytest.approx(
        np.std(model.wavefront_nm(wave)[model.pupil]))
    np.testing.assert_array_equal(opd_to_surface_nm(wave), wave / 2)


@pytest.mark.parametrize("n,oversampling", [(12, 1), (12, 2), (13, 2), (13, 3)])
def test_fft_field_matches_direct_matrix_fourier_sum(n, oversampling):
    model = OpticalModel(OpticsConfig(pupil_pixels=n, oversampling=oversampling))
    wave = np.array([31, -11, 7, 14, -5, 20])
    pupil = model.pupil_field(wave)
    m = n * oversampling
    coordinates = np.arange(n) + (m - n)//2 - m//2
    frequencies = np.arange(m) - m//2
    matrix = np.exp(-2j * np.pi * np.outer(frequencies, coordinates) / m) / np.sqrt(m)
    independent = matrix @ pupil @ matrix.T
    np.testing.assert_allclose(model.field(wave), independent, rtol=2e-13, atol=2e-15)


def test_opposite_defocus_ambiguity_and_probe_disambiguation(model):
    positive = np.array([45., 0, 0, 0, 0, 0])
    negative = -positive
    np.testing.assert_allclose(model.intensity(positive), model.intensity(negative), atol=1e-16)
    probe = np.array([80., 0, 0, 0, 0, 0])
    assert np.linalg.norm(model.intensity(positive-probe)-model.intensity(negative-probe)) > 0.02
    assert model.evaluate(positive-positive)["residual_wfe_nm"] == 0
    assert model.evaluate(negative-positive)["residual_wfe_nm"] == pytest.approx(90)


@pytest.mark.parametrize("bad", [[0]*5, [0]*7, [0, 0, 0, 0, 0, np.nan], [np.inf]*6])
def test_reject_invalid_coefficients(bad):
    with pytest.raises(ValueError): coefficients(bad)


@pytest.mark.parametrize("kw", [{"pupil_pixels": 11}, {"oversampling": 1.2},
                               {"wavelength_nm": 0}, {"wavelength_nm": float("nan")}])
def test_reject_invalid_optical_config(kw):
    with pytest.raises(ValueError): OpticsConfig(**kw)
