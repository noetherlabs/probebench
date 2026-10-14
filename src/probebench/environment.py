"""Budgeted camera observations; hidden state stays in the driver/evaluator."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .optics import OpticalModel, coefficients, opd_to_surface_nm


@dataclass(frozen=True)
class EpisodeConfig:
    max_exposures: int = 3
    photon_budget: int = 600_000
    photons_per_exposure: int = 200_000
    read_noise_electrons: float = 1.0
    command_limit_nm: float = 200.0

    def __post_init__(self):
        for name in ("max_exposures", "photon_budget", "photons_per_exposure"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.read_noise_electrons) or self.read_noise_electrons < 0:
            raise ValueError("read noise must be finite and nonnegative")
        if not np.isfinite(self.command_limit_nm) or self.command_limit_nm <= 0:
            raise ValueError("command limit must be finite and positive")


@dataclass(frozen=True)
class Observation:
    index: int
    applied_opd_nm: tuple[float, ...]
    expected_photons: int
    cumulative_expected_photons: int
    image_electrons: np.ndarray

    def metadata(self):
        return {
            "index": self.index,
            "applied_opd_nm": list(self.applied_opd_nm),
            "expected_photons": self.expected_photons,
            "cumulative_expected_photons": self.cumulative_expected_photons,
        }


def display_image(observation: Observation):
    """Public fixed scaling; never use image extrema or evaluator information."""
    return np.log1p(np.maximum(observation.image_electrons, 0)) / np.log1p(
        observation.expected_photons
    )


class Episode:
    """Private simulation driver. Policies receive Observations, never this object."""

    def __init__(self, model, config, hidden_opd_nm, detector_seed):
        self.model: OpticalModel = model
        self.config: EpisodeConfig = config
        self._hidden = coefficients(hidden_opd_nm)
        self._rng = np.random.default_rng(detector_seed)
        self._records = []
        self.observations = []
        self._photons = 0
        self._closed = False
        self._applied = np.zeros(6)

    def _validate_command(self, command):
        command = coefficients(command)
        if np.any(np.abs(command) > self.config.command_limit_nm):
            raise ValueError("Command exceeds per-mode OPD coefficient limit")
        return command

    def expose(self, command_opd_nm, expected_photons=None):
        if self._closed:
            raise RuntimeError("Episode is already submitted")
        command = self._validate_command(command_opd_nm)
        photons = self.config.photons_per_exposure if expected_photons is None else expected_photons
        if type(photons) is not int or photons <= 0:
            raise ValueError("Exposure photons must be a positive integer")
        if len(self.observations) >= self.config.max_exposures:
            raise ValueError("Exposure budget exhausted")
        if self._photons + photons > self.config.photon_budget:
            raise ValueError("Photon budget exhausted")
        # Validation precedes both detector draws and accounting mutations.
        residual = self._hidden - command
        mean_counts = photons * self.model.intensity(residual)
        photoelectrons = self._rng.poisson(mean_counts)
        measured = photoelectrons.astype(np.float64) + self._rng.normal(
            0, self.config.read_noise_electrons, mean_counts.shape
        )
        measured.setflags(write=False)
        self._photons += photons
        self._applied = command.copy()
        observation = Observation(
            len(self.observations), tuple(command), photons, self._photons, measured
        )
        self.observations.append(observation)
        self._records.append({
            **observation.metadata(),
            **self.model.evaluate(residual),
            "realized_photoelectrons": int(photoelectrons.sum()),
        })
        return observation

    def submit(self, command_opd_nm):
        if self._closed:
            raise RuntimeError("Episode is already submitted")
        command = self._validate_command(command_opd_nm)
        if not self.observations:
            raise ValueError("At least one observation is required")
        self._applied = command.copy()
        self._closed = True
        return self.evaluator_record()

    def evaluator_record(self):
        if not self._closed:
            raise RuntimeError("Final evaluator record requires submission")
        metrics = self.model.evaluate(self._hidden - self._applied)
        initial = self.model.evaluate(self._hidden)
        measured_indices = [o.index for o in self.observations
                            if np.array_equal(self._applied, o.applied_opd_nm)]
        return {
            "hidden_opd_nm": self._hidden.tolist(),
            "initial": initial,
            "final": metrics,
            "final_applied_opd_nm": self._applied.tolist(),
            "final_mirror_surface_coefficients_nm": opd_to_surface_nm(self._applied).tolist(),
            "final_command_has_camera_observation": bool(measured_indices),
            "final_command_observation_indices": measured_indices,
            "final_command_is_last_observation": bool(np.array_equal(
                self._applied, self.observations[-1].applied_opd_nm
            )),
            "expected_photons_used": self._photons,
            "exposures": len(self.observations),
            "trajectory": self._records.copy(),
            "peak_probe_wfe_increase_nm": max(
                0.0, max(r["residual_wfe_nm"] for r in self._records) - initial["residual_wfe_nm"]
            ),
        }
