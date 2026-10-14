"""Independent scalar Fourier reference with explicit OPD nanometre units."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

MODES = ("defocus", "astigmatism_cos", "astigmatism_sin", "coma_x", "coma_y", "spherical")


def coefficients(value) -> NDArray[np.float64]:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError("Expected six finite OPD coefficients in nanometres")
    return result


def opd_to_surface_nm(opd_nm):
    """Reflective, normally incident ideal mirror: OPD = 2 * surface height."""
    return np.asarray(opd_nm, dtype=np.float64) / 2


@dataclass(frozen=True)
class OpticsConfig:
    pupil_pixels: int = 32
    oversampling: int = 2
    wavelength_nm: float = 550.0

    def __post_init__(self):
        if type(self.pupil_pixels) is not int or self.pupil_pixels < 12:
            raise ValueError("pupil_pixels must be an integer >= 12")
        if type(self.oversampling) is not int or self.oversampling < 1:
            raise ValueError("oversampling must be a positive integer")
        if not np.isfinite(self.wavelength_nm) or self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be finite and positive")


class OpticalModel:
    """Known calibration model; this object never stores the unknown aberration."""

    def __init__(self, config: OpticsConfig):
        self.config = config
        n = config.pupil_pixels
        x = 2 * (np.arange(n, dtype=np.float64) - (n - 1) / 2) / n
        xx, yy = np.meshgrid(x, x)
        r2 = xx**2 + yy**2
        self.pupil = r2 <= 1
        # Analytic real Zernikes, then finite-pupil piston removal and RMS scaling.
        raw = np.stack((
            np.sqrt(3) * (2 * r2 - 1),
            np.sqrt(6) * (xx**2 - yy**2),
            np.sqrt(6) * 2 * xx * yy,
            np.sqrt(8) * (3 * r2 - 2) * xx,
            np.sqrt(8) * (3 * r2 - 2) * yy,
            np.sqrt(5) * (6 * r2**2 - 6 * r2 + 1),
        ))
        raw -= raw[:, self.pupil].mean(axis=1)[:, None, None]
        raw /= np.sqrt(np.mean(raw[:, self.pupil] ** 2, axis=1))[:, None, None]
        self.basis = raw * self.pupil
        self.amplitude = self.pupil.astype(np.float64) / np.sqrt(self.pupil.sum())
        self.ideal_peak = float(self.intensity(np.zeros(6)).max())
        self.pupil.setflags(write=False)
        self.basis.setflags(write=False)
        self.amplitude.setflags(write=False)

    def wavefront_nm(self, opd_coefficients_nm):
        return np.einsum("m,mij->ij", coefficients(opd_coefficients_nm), self.basis)

    def pupil_field(self, residual_opd_nm):
        phase = 2 * np.pi * self.wavefront_nm(residual_opd_nm) / self.config.wavelength_nm
        return self.amplitude * np.exp(1j * phase)

    def field(self, residual_opd_nm):
        pupil = self.pupil_field(residual_opd_nm)
        n = self.config.pupil_pixels
        padded_n = n * self.config.oversampling
        before = (padded_n - n) // 2
        after = padded_n - n - before
        padded = np.pad(pupil, ((before, after), (before, after)))
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded), norm="ortho"))

    def intensity(self, residual_opd_nm):
        # No image-wise renormalization: discrete Parseval power is already one.
        return np.abs(self.field(residual_opd_nm)) ** 2

    def intensity_and_jacobian(self, residual_opd_nm):
        """Intensity and analytic derivatives per nanometre of each OPD coefficient."""
        pupil = self.pupil_field(residual_opd_nm)
        derivative = (2j * np.pi / self.config.wavelength_nm) * self.basis * pupil
        fields = np.concatenate((pupil[None], derivative), axis=0)
        n = self.config.pupil_pixels
        m = n * self.config.oversampling
        before, after = (m - n) // 2, m - n - (m - n) // 2
        padded = np.pad(fields, ((0, 0), (before, after), (before, after)))
        focal = np.fft.fftshift(np.fft.fft2(
            np.fft.ifftshift(padded, axes=(-2, -1)), axes=(-2, -1), norm="ortho"
        ), axes=(-2, -1))
        return np.abs(focal[0])**2, 2 * np.real(focal[0].conj() * focal[1:])

    def evaluate(self, residual_opd_nm):
        wavefront = self.wavefront_nm(residual_opd_nm)[self.pupil]
        piston_removed = wavefront - wavefront.mean()
        image = self.intensity(residual_opd_nm)
        return {
            "residual_wfe_nm": float(np.sqrt(np.mean(piston_removed**2))),
            "strehl_peak_ratio": float(image.max() / self.ideal_peak),
        }
