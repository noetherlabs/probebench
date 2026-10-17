"""Versioned public messages for trusted local optical-calibration agents."""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
import re
import secrets

import matplotlib.image
import numpy as np

from .budget_policies import ClassicalPolicy, PolicyConfig, visible_history_digest
from .environment import display_image
from .optics import MODES

PROTOCOL = "probebench.agent.v1"
MAX_ACTION_BYTES = 16_384
TRACKS = ("direct_action", "tool_assisted")
ABORT_REASONS = frozenset({"input_eof", "input_timeout", "input_too_large", "request_limit"})


def encode_message(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_key")
        value[key] = item
    return value


def _nonfinite_constant(_):
    raise ValueError("nonfinite_json")


def parse_action(raw):
    """Reject ambiguous JSON before interpreting any action or touching the camera."""
    try:
        payload = raw.encode("utf-8") if isinstance(raw, str) else raw
        if not isinstance(payload, bytes) or len(payload) > MAX_ACTION_BYTES:
            raise ValueError("invalid_json")
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=_nonfinite_constant)
    except (UnicodeError, ValueError, RecursionError, TypeError):
        raise ValueError("invalid_json") from None
    if not isinstance(value, dict):
        raise ValueError("invalid_schema")
    return value


def public_observation(observation):
    buffer = io.BytesIO()
    matplotlib.image.imsave(buffer, display_image(observation), format="png", origin="lower",
                           cmap="magma", vmin=0, vmax=1)
    return {**observation.metadata(), "image_electrons": observation.image_electrons.tolist(),
            "image_png_base64": base64.b64encode(buffer.getvalue()).decode("ascii")}


def validate_limits(track, max_requests, max_tool_calls, max_mle_iterations):
    if track not in TRACKS:
        raise ValueError("Unknown agent track")
    if type(max_requests) is not int or max_requests < 1:
        raise ValueError("Positive integer request limit required")
    if type(max_tool_calls) is not int or max_tool_calls < 0:
        raise ValueError("Nonnegative integer tool-call limit required")
    if (track == "direct_action" and max_tool_calls != 0
        or track == "tool_assisted" and max_tool_calls < 1):
        raise ValueError("Tool-call allowance must agree with the track")
    if type(max_mle_iterations) is not int or max_mle_iterations < 1:
        raise ValueError("Positive integer solver allowance required")


class AgentSession:
    """Own private simulation state; serialize only an explicit public allowlist.

    This class and its checkpoint methods are trusted driver code, not a Python sandbox.
    Agent input enters only through handle(); evaluator_record() is never sent to it.
    """

    def __init__(self, episode, *, track, max_requests=128, max_tool_calls=0,
                 max_mle_iterations=200, nonce=None):
        validate_limits(track,max_requests,max_tool_calls,max_mle_iterations)
        if episode.observations or episode._closed:
            raise ValueError("Agent sessions require a fresh episode")
        nonce = secrets.token_hex(16) if nonce is None else nonce
        if not isinstance(nonce, str) or re.fullmatch(r"[0-9a-f]{32}", nonce) is None:
            raise ValueError("Invalid session nonce")
        self._episode = episode
        self._policy = ClassicalPolicy("phase_diversity_mle_v1", episode.model, episode.config,
            PolicyConfig(max_mle_iterations=max_mle_iterations))
        self.track, self.nonce = track, nonce
        self.max_requests, self.max_tool_calls = max_requests, max_tool_calls
        self.requests_used, self.tool_calls_used, self._revision = 0, 0, 0
        self.status, self.failure_reason = "active", None
        self._error, self._tool_result, self._submitted = None, None, None
        self._observation = public_observation(episode.expose(np.zeros(6)))

    @property
    def request_id(self):
        # The nonce is independent of wavefronts, detector randomness and fixture IDs.
        return hashlib.sha256(f"{self.nonce}:{self._revision}".encode()).hexdigest()

    def _has_diversity(self):
        history = self._episode.observations
        return len(history) >= 2 and any(o.applied_opd_nm != history[0].applied_opd_nm for o in history)

    def request(self):
        instrument = self._episode.config
        exposures = len(self._episode.observations)
        photons = self._episode.observations[-1].cumulative_expected_photons
        available = []
        if self.status == "active":
            available = ["submit"]
            if exposures < instrument.max_exposures and photons+instrument.photons_per_exposure <= instrument.photon_budget:
                available.append("expose")
            if (self.track == "tool_assisted" and self.tool_calls_used < self.max_tool_calls
                and self._has_diversity()):
                available.append("estimate")
        measured = None if self._submitted is None else [o.index for o in self._episode.observations
            if np.array_equal(o.applied_opd_nm,self._submitted)]
        return deepcopy({
            "protocol":PROTOCOL, "kind":"decision" if self.status == "active" else "final",
            "request_id":self.request_id, "track":self.track,
            "estimator":None if self.track == "direct_action" else {
                "method":"phase_diversity_mle_v1", "objective":"poisson_gaussian_nll",
                "max_iterations_per_start":self._policy.config.max_mle_iterations,
                "observations":"all acquired frames", "minimum_distinct_commands":2,
                "coefficient_bound_opd_nm":instrument.command_limit_nm,
                "initial_defocus_opd_nm":[0,min(50.,instrument.command_limit_nm/2),
                                            -min(50.,instrument.command_limit_nm/2)]},
            "instrument":{"optics":asdict(self._episode.model.config), **asdict(instrument),
                "modes":list(MODES), "command_units":"opd_nm", "mirror_surface_nm_per_opd_nm":0.5,
                "image_units":"electrons", "image_array_row_zero":"bottom",
                "display":{"transform":"log1p(max(electrons,0))/log1p(expected_photons)",
                           "minimum":0, "maximum":1, "colormap":"magma"}},
            "resources":{"exposures_used":exposures, "expected_photons_used":photons,
                "exposures_remaining":instrument.max_exposures-exposures,
                "expected_photons_remaining":instrument.photon_budget-photons,
                "requests_used":self.requests_used, "requests_remaining":self.max_requests-self.requests_used,
                "tool_calls_used":self.tool_calls_used, "tool_calls_remaining":self.max_tool_calls-self.tool_calls_used},
            "available_actions":available, "latest_observation":self._observation,
            "tool_result":self._tool_result, "error":self._error,
            "status":self.status, "failure_reason":self.failure_reason,
            "submitted_command_opd_nm":self._submitted, "submitted_command_observation_indices":measured,
        })

    def _validate(self, raw):
        action = parse_action(raw)
        operation = action.get("action")
        if not isinstance(operation,str) or operation not in {"expose","submit","estimate"}:
            raise ValueError("invalid_schema")
        fields = {"protocol","request_id","action"}
        if operation != "estimate":
            fields.add("command_opd_nm")
        if set(action) != fields or action["protocol"] != PROTOCOL:
            raise ValueError("invalid_schema")
        if action["request_id"] != self.request_id:
            raise ValueError("stale_request")
        if operation == "estimate":
            if self.track != "tool_assisted" or self.tool_calls_used >= self.max_tool_calls:
                raise ValueError("action_unavailable")
            if not self._has_diversity():
                raise ValueError("diverse_measurements_required")
        else:
            if operation not in self.request()["available_actions"]:
                raise ValueError("action_unavailable")
            command = action["command_opd_nm"]
            if (not isinstance(command,list) or len(command) != 6
                or any(type(v) not in (int,float) for v in command)):
                raise ValueError("invalid_command")
            try:
                finite = np.isfinite(np.array(command,dtype=float)).all()
            except (ValueError,OverflowError):
                raise ValueError("invalid_command") from None
            if not finite:
                raise ValueError("invalid_command")
            if any(abs(v) > self._episode.config.command_limit_nm for v in command):
                raise ValueError("command_out_of_bounds")
        return action

    def handle(self, raw, *, retained_fits=None):
        """Consume one bounded request; rejected actions never draw camera randomness.

        retained_fits is a trusted checkpoint input, never part of the JSON protocol.
        Invalid requests consume the separately declared request allowance.
        """
        if self.status != "active":
            raise RuntimeError("Agent session is closed")
        self.requests_used += 1
        try:
            action = self._validate(raw)
        except ValueError as error:
            self._error = str(error)
        else:
            self._error = None
            self._tool_result = None
            operation = action["action"]
            if operation == "expose":
                self._observation = public_observation(self._episode.expose(action["command_opd_nm"]))
            elif operation == "submit":
                self._episode.submit(action["command_opd_nm"])
                self._submitted = list(action["command_opd_nm"])
                self.status = "submitted"
            else:
                history = self._episode.observations
                key = visible_history_digest(history)
                cached = any(f["visible_history_sha256"] == key for f in self._policy.fit_records)
                if retained_fits is not None:
                    before = self._policy.fit_records
                    if retained_fits[:len(before)] != before:
                        raise ValueError("Changed completed estimator records")
                    self._policy.restore_fit_records(retained_fits,history)
                    if not any(f["visible_history_sha256"] == key for f in self._policy.fit_records):
                        raise ValueError("Missing retained estimator fit")
                estimate = self._policy.final_command(history)
                self.tool_calls_used += 1
                self._tool_result = {"estimated_opd_nm":estimate.tolist(), "cached":cached,
                                     "diagnostics":deepcopy(self._policy.fit_records[-1])}
            self._revision += 1
        if self.status == "active" and self.requests_used >= self.max_requests:
            self.abort("request_limit")
        return self.request()

    def abort(self, reason):
        if reason not in ABORT_REASONS or self.status != "active":
            raise ValueError("Invalid termination")
        self._submitted = list(self._episode.observations[-1].applied_opd_nm)
        self._episode.submit(self._submitted)
        self.status, self.failure_reason = "failed", reason
        self._revision += 1
        return self.request()

    def retained_fits(self):
        return deepcopy(self._policy.fit_records)

    def evaluator_record(self):
        if self.status == "active":
            raise RuntimeError("Final evaluation requires a closed session")
        return {**self._episode.evaluator_record(), "protocol":PROTOCOL, "track":self.track,
                "status":self.status, "failure_reason":self.failure_reason,
                "requests_used":self.requests_used, "tool_calls_used":self.tool_calls_used,
                "fit_records":self.retained_fits()}
