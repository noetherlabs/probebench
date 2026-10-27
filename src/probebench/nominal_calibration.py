"""Nominal-command observations for a private scalar command-response mismatch.

The public calibration predicts unit OPD response. The actual response belongs
only to the simulation driver and evaluator; this is not a process sandbox.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from numbers import Real

import numpy as np

from .budget_policies import ClassicalPolicy, PolicyConfig
from .environment import Episode, EpisodeConfig, Observation
from .optics import coefficients

OBSERVATION_SCHEMA = "probebench.nominal-observation/1"
EVALUATION_SCHEMA = "probebench.scalar-gain-evaluation/1"
RUN_SCHEMA = "probebench.nominal-classical-episode/1"


class NumericalObservationError(RuntimeError):
    """A produced camera observation failed validation; the episode is terminal."""


def _nominal_calibration():
    return {"response": "identity", "command_units": "nm_nominal_opd",
            "prediction_units": "nm_opd"}


def _command_coefficients(value):
    if np.iscomplexobj(value):
        raise ValueError("Nominal command coefficients must be real")
    try:
        return coefficients(value)
    except (TypeError, OverflowError) as error:
        raise ValueError("Expected six finite nominal OPD coefficients") from error


def _immutable_image(image):
    array = np.asarray(image)
    if (array.ndim != 2 or not array.size or not np.issubdtype(array.dtype, np.number)
            or np.iscomplexobj(array)):
        raise ValueError("Expected a nonempty real detector image")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("Expected finite detector values")
    return _image_snapshot(array)


def _image_snapshot(array):
    # A bytes-backed copy cannot be made writable with ndarray.setflags().
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True)
class NominalObservation:
    """A detached measurement plus requested command, never measured actuation.

    Commands are in nominal OPD nanometres. Unit nominal calibration predicts
    the same OPD; the actual optical response is deliberately not a field.
    """

    index: int
    commanded_opd_nm: tuple[float, ...]
    expected_photons: int
    cumulative_expected_photons: int
    image_electrons: np.ndarray

    def __post_init__(self):
        if type(self.index) is not int or self.index < 0:
            raise ValueError("Observation index must be a nonnegative integer")
        for name in ("expected_photons", "cumulative_expected_photons"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("Photon counters must be positive integers")
        if self.cumulative_expected_photons < self.expected_photons:
            raise ValueError("Cumulative photons cannot be less than this exposure")
        object.__setattr__(self, "commanded_opd_nm", tuple(_command_coefficients(self.commanded_opd_nm)))
        object.__setattr__(self, "image_electrons", _immutable_image(self.image_electrons))

    def metadata(self):
        """Fresh JSON-compatible public fields with explicit nominal semantics."""
        return {"schema": OBSERVATION_SCHEMA, "index": self.index,
                "commanded_opd_nm": list(self.commanded_opd_nm),
                "nominal_applied_opd_nm": list(self.commanded_opd_nm),
                "nominal_calibration": _nominal_calibration(),
                "expected_photons": self.expected_photons,
                "cumulative_expected_photons": self.cumulative_expected_photons}


class GainEpisode:
    """Private driver for ``residual = hidden_opd - command_gain * command``.

    ``config.command_limit_nm`` bounds nominal commands. The internal camera's
    physical envelope is gain times that bound; there is no extra clipping.
    Gain is positive, constant within the episode, and evaluator-only.
    """

    def __init__(self, model, config: EpisodeConfig, hidden_opd_nm, detector_seed,
                 *, command_gain: float):
        if isinstance(command_gain, (bool, np.bool_)) or not isinstance(command_gain, Real):
            # All invalid gain specifications use the same input-validation error.
            raise ValueError("command_gain must be a finite positive real scalar")  # noqa: TRY004
        try:
            self._gain = float(command_gain)
        except (ValueError, OverflowError) as error:
            raise ValueError("command_gain is not representable as float64") from error
        if not np.isfinite(self._gain) or self._gain <= 0:
            raise ValueError("command_gain must be a finite positive real scalar")
        physical_limit = self._gain * config.command_limit_nm
        if not np.isfinite(physical_limit) or physical_limit <= 0:
            raise ValueError("Scaled physical command limit is not representable")
        self._nominal_config = config
        self._episode = Episode(model, replace(config, command_limit_nm=physical_limit),
                                hidden_opd_nm, detector_seed)
        self._observations = []
        self._final_command = None
        self._failure = None
        self._failed_observation = None

    @property
    def observations(self):
        """Immutable public snapshots; no reference to the internal camera state."""
        return tuple(self._observations)

    @property
    def failed_observation(self):
        """Private physical measurement retained on failure, including nonfinite pixels.

        This detached immutable snapshot is evaluator evidence, never policy input.
        It is ``None`` unless a produced observation failed public validation.
        """
        return self._failed_observation

    def _command(self, value):
        if self._failure is not None:
            raise RuntimeError("Episode is terminal after a numerical observation failure")
        if self._final_command is not None:
            raise RuntimeError("Episode is already submitted")
        command = _command_coefficients(value)
        if np.any(np.abs(command) > self._nominal_config.command_limit_nm):
            raise ValueError("Command exceeds nominal per-mode OPD coefficient limit")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            physical = self._gain * command
        if (not np.isfinite(physical).all()
                or np.any((command != 0) & (physical == 0))):
            raise ValueError("Physical command is not representable")
        return command, physical

    def expose(self, commanded_opd_nm, expected_photons=None):
        command, physical = self._command(commanded_opd_nm)
        measured = self._episode.expose(physical, expected_photons)
        try:
            public = NominalObservation(measured.index, tuple(command), measured.expected_photons,
                                        measured.cumulative_expected_photons, measured.image_electrons)
        except ValueError as error:
            # The camera has consumed the draw and charged the exposure. Preserve
            # that evidence and stop; neither retry nor a final score is valid.
            self._failure = {"stage": "public_observation_validation",
                             "reason": str(error), "observation_index": measured.index,
                             "commanded_opd_nm": command.tolist()}
            self._failed_observation = Observation(
                measured.index, measured.applied_opd_nm, measured.expected_photons,
                measured.cumulative_expected_photons, _image_snapshot(measured.image_electrons))
            raise NumericalObservationError("Produced camera observation is invalid; episode is terminal") from error
        self._observations.append(public)
        return public

    def _policy_history(self):
        """Internal nominal-model projection for the unchanged classical solver.

        Legacy ``applied_opd_nm`` means the OPD assumed by this solver, not an
        actuation measurement. This projection is never the public new schema.
        """
        return tuple(Observation(o.index, o.commanded_opd_nm, o.expected_photons,
                                 o.cumulative_expected_photons, o.image_electrons)
                     for o in self._observations)

    def submit(self, commanded_opd_nm):
        command, physical = self._command(commanded_opd_nm)
        self._episode.submit(physical)
        self._final_command = tuple(command)
        return self.evaluator_record()

    def evaluator_record(self):
        """Private evidence, including actual actuation and legacy physical metrics."""
        if self._failure is None:
            physical = self._episode.evaluator_record()
        else:
            physical = {"hidden_opd_nm": self._episode._hidden.tolist(),
                        "trajectory": deepcopy(self._episode._records),
                        "exposures": len(self._episode.observations),
                        "expected_photons_used": sum(o.expected_photons
                                                    for o in self._episode.observations)}
        result = {"schema": EVALUATION_SCHEMA, "command_gain": self._gain,
                "status": "numerical_failure" if self._failure is not None else "submitted",
                "nominal_calibration": _nominal_calibration(),
                "nominal_command_limit_nm": self._nominal_config.command_limit_nm,
                "nominal_observations": [o.metadata() for o in self._observations],
                "final_commanded_opd_nm": (None if self._final_command is None
                                           else list(self._final_command)),
                "physical": deepcopy(physical)}
        if self._failure is not None:
            result["failure"] = {**deepcopy(self._failure),
                                 "public_observation_count": len(self._observations),
                                 "produced_observation_count": len(self._episode.observations),
                                 "failed_physical_observation": self._failed_observation.metadata(),
                                 "nonfinite_pixel_count": int(np.count_nonzero(
                                     ~np.isfinite(self._failed_observation.image_electrons)))}
        return result


def run_classical_episode(model, config: EpisodeConfig, hidden_opd_nm, detector_seed,
                          *, command_gain, method, policy_config=None, policy_seed=0):
    """Run one in-memory classical episode under nominal unit calibration.

    Returned observations are public; ``evaluation`` is private evaluator data.
    This helper selects no cases, writes no artifacts, and supplies no source or
    cohort provenance. Callers must provide those before an experimental study.
    """
    if config.photon_budget < config.max_exposures * config.photons_per_exposure:
        raise ValueError("Full declared exposure count exceeds the photon budget")
    if policy_config is not None and not isinstance(policy_config, PolicyConfig):
        raise ValueError("policy_config must be a PolicyConfig")
    settings = policy_config or PolicyConfig()
    episode = GainEpisode(model, config, hidden_opd_nm, detector_seed, command_gain=command_gain)
    # Only the original nominal instrument and public optical model reach policy.
    policy = ClassicalPolicy(method, model, config, settings, policy_seed)
    try:
        for _ in range(config.max_exposures):
            command = policy.command(episode._policy_history())
            episode.expose(command)
    except NumericalObservationError:
        evaluation = episode.evaluator_record()
    else:
        evaluation = episode.submit(policy.final_command(episode._policy_history()))
    return {"schema": RUN_SCHEMA, "method": method, "policy_configuration": asdict(settings),
            "observations": episode.observations, "evaluation": evaluation,
            "failed_observation": episode.failed_observation,
            "fit_records": deepcopy(policy.fit_records)}
