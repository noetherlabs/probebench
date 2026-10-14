# Local agent protocol

`probebench.agent.v1` is a JSON-lines interface to the static six-mode optical
environment. The driver owns the camera, budgets and hidden state. A trusted client
receives public messages on standard output and sends one action per line on standard
input. The driver does not invoke a model API or execute client-supplied code.

## Information and actions

The first request contains an acquired zero-command image. It costs one exposure and
the configured expected photons per exposure. Each request carries the public optical
and detector calibration, mode order, command bounds, resource counters, current request
identifier, available actions and latest observation. Clients retain their earlier
observations; the driver does not resend the entire image history on every turn.

The observation includes signed float64 electron values as JSON numbers and a base64
PNG display. The display uses `log1p(max(electrons,0))/log1p(expected_photons)`, magma and
fixed limits 0–1. Array row zero is the bottom image row. Display clipping does not
alter the raw values. There is no per-image extrema normalization.

| Track | Actions |
| --- | --- |
| `direct_action` | Apply an absolute six-mode OPD command and expose; submit a final command. |
| `tool_assisted` | The same actions, plus a separately budgeted phase-diversity estimator using all acquired frames. |

The supplied estimator is the validated Poisson–Gaussian multistart fit. Its start
offsets, coefficient bound and per-start iteration allowance are disclosed in the
request. Its result contains the estimate, every optimizer-start diagnostic, convergence
or numerical-failure status and whether the result was cached. A numerical failure
retains the measured-camera-score fallback. No evaluator quantity enters the fit.

An estimator call requires at least two distinct applied commands. This is a numerical
precondition, not proof of phase identifiability. Distinct commands can still leave
ambiguities, and convergence does not establish a correct correction. Repeated calls
on unchanged observations reuse the saved fit but consume the declared tool-call
allowance. The initial request does not advertise the estimator as available until
its measurement precondition is met.

The driver never includes latent wavefront coefficients, detector seeds, evaluator WFE,
Strehl, realized photon counts or fixture identity in a request. These values remain in
the separate retained experiment files. Track restrictions govern the tools exposed by
this interface; they cannot constrain computations performed independently by a client.

## Action format

Copy the current request identifier into every action. For example, an exposure action
has exactly these fields, with the placeholder replaced by the received identifier:

```json
{"protocol":"probebench.agent.v1","request_id":"<current request identifier>","action":"expose","command_opd_nm":[80,0,0,0,0,0]}
```

Use `"action":"submit"` with the same command field for a final submission.
Use `"action":"estimate"` with only `protocol`, `request_id` and `action` to request
the assisted estimator. Unknown fields are rejected. Commands are absolute OPD
coefficients in nanometres, ordered as disclosed by the instrument. Residual phase is
incident OPD minus applied OPD. Reflective mirror-surface displacement is half the OPD.

The parser accepts at most 16,384 UTF-8 bytes per action, including its newline. It
rejects duplicate JSON keys, nonfinite numbers, boolean or string coefficients,
incorrect vector lengths and out-of-bounds commands. A repeated identifier after a
successful action is stale. Rejected actions leave the camera RNG, exposure count and
photon count unchanged. They consume the separate request allowance and keep the
current identifier available for a corrected action.

All exposures have the configured photon count. An unavailable exposure or estimator
call is rejected before execution. Submission adds no camera image or photons. The
final acknowledgement lists earlier observations, if any, of the submitted command.
Its latest image remains the last acquired image; it is not a simulated image of an
unmeasured final correction. Final WFE and Strehl are absent from the acknowledgement.

## Termination and records

EOF, input timeout, oversized input or exhaustion of the request allowance closes a
failed episode at its last measured command. The driver retains this outcome and its
measurements. A failed session must not be silently counted as a successful submission
or excluded from a later benchmark analysis.

The request counter counts bounded action messages passed to the protocol parser.
Transport termination events are retained separately; they do not represent another
parsed action or camera exposure.

The input deadline covers partial-line stalls as well as an entirely absent response.
It does not impose a hard wall-clock limit on the numerical optimizer. Each journal
record separately measures input-wait time and driver-processing time. Fit records
retain estimator duration. These are local execution measurements, not model API
expenditure, complete decision latency, or real-time control guarantees.

Before acquisition, the driver freezes configuration, source and numerical runtime.
The session additionally binds the PNG renderer, Pillow and zlib versions. It publishes
the received input before applying it, then publishes the complete response and saved
fits before sending the response to the client. A completed session contains:

| Artifact | Contents |
| --- | --- |
| `study.json`, `source/` | Configuration, numerical runtime and source snapshot |
| `session.json` | Initial public request, independent session nonce and display runtime |
| `turns/turn_*.json` | Hash-chained inputs, public responses, saved fits and separate durations |
| `evaluator.json` | Final hidden-state evaluation, resource accounting and outcome status |
| `manifest.json` | Complete artifact membership and hashes |

Raw signed images are retained losslessly in public observation JSON. Each turn records
the exact incoming bytes in base64, including rejected input. Logs can therefore contain
sensitive client text and require review before distribution.

## Execution and resumption

The configuration is a driver input. Do not send its contents or filename to a model.
Only standard-output protocol messages belong on the agent-facing channel; standard
error is a separate operator diagnostic channel.

```sh
uv run --locked python -m probebench.agent_session \
  --config configs/agent_protocol_development.json --output work/agent-session
```

Run the supplied scripted transport control through the same process interface:

```sh
uv run --locked python scripts/run_agent_protocol_control.py \
  --config configs/agent_protocol_development.json --output work/protocol-control
uv run --locked python -m probebench.agent_session --verify work/protocol-control
```

The control applies a known defocus probe and uses the estimator in the assisted track.
In the direct track it submits zero after probing. It is a transport qualification
program, not a language model or a benchmark of agent capability.

Repeat either execution command with `--resume` after an interruption. The driver
requires identical source, configuration and runtime. It restores camera state by
deterministic replay, checks every saved public response, and reuses completed fits
without optimizing again. A saved input whose response was not published is recovered
before another decision is requested. A completed response is reissued with the same
identifier, so repeating its earlier action cannot acquire another exposure.

Resumption preserves the physical observation record. It does not restore an external
model's conversation or sampler state; the client owns that state. Verification checks
saved-fit bindings and response consistency but does not independently rerun each
optimization. A completed session's evaluator is recomputed from its retained actions
and wavefront. Output directories are exclusively locked and completed records are
not replaced.

The supplied runner accepts explicitly labelled development configurations. It does
not define a reserved cohort, population uncertainty analysis or model ranking.
See the [security contract](../SECURITY.md): standard I/O and request identifiers are
not authentication or isolation from an agent with filesystem/process access.
