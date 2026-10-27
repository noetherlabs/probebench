# Nominal calibration and scalar command response

`probebench.nominal_calibration` separates a requested command from its actual
optical response. It provides an opt-in measurement contract and an in-memory
classical episode runner; it does not provide a new experimental result.

Let `a` denote the incident six-mode OPD coefficients, `u` the requested command,
and `g > 0` a scalar response fixed for the duration of the episode. The private
plant applies `g*u`, giving residual OPD `B @ (a - g*u)`. The policy's nominal
calibration predicts `B @ u`. Coefficients are in nanometres of OPD; a reflective
surface displacement at normal incidence is half the actual OPD.

This model changes the command-to-phase response, not detector gain, photon flux,
the aberration basis or the wavefront during the episode. Scalar response does
not represent actuator coupling, misregistration, hysteresis or changing gain.

## Observation contract

`GainEpisode.expose(u)` returns `NominalObservation`. Its metadata schema is
`probebench.nominal-observation/1` and contains:

| Field | Meaning |
| --- | --- |
| `commanded_opd_nm` | Requested coefficients in nominal OPD nanometres. |
| `nominal_applied_opd_nm` | OPD predicted by the declared unit nominal response. |
| `nominal_calibration` | Identity response, command units and prediction units. |
| `index` | Zero-based exposure index. |
| `expected_photons` | Expected incident photons for this exposure. |
| `cumulative_expected_photons` | Expected incident photons charged so far. |

The object also carries the signed detector array in `image_electrons`. These
values are copied into an immutable byte-backed NumPy array; changing a returned
metadata dictionary cannot change stored observations. Public metadata does not
contain the true gain, actual OPD, truth coefficients, detector seed or evaluator
metrics. Image differences caused by the physical response are intentional.

The existing `environment.Observation` schema is unchanged. Inside the classical
runner, a compatibility projection supplies **nominal predicted OPD** in the
legacy estimator's `applied_opd_nm` field. That projection is the solver's assumed
model input, not a measured-actuation record, and is not serialized as the new
public observation schema. The estimator and classical policies are unchanged.

## Actions and evaluation

`EpisodeConfig.command_limit_nm` bounds requested nominal coefficients. The
private camera's physical envelope is `g*command_limit_nm`; the adapter does not
silently clip the scaled command. This is a mathematical envelope, not a measured
hardware stroke limit. Gain and the scaled limit must be finite, positive and
representable. Invalid commands and exhausted budgets are rejected before
camera randomness or exposure accounting advances.

Final submission uses the same response `g*u`. The returned private evaluation
has schema `probebench.scalar-gain-evaluation/1` and status `submitted`. It includes the true gain,
nominal commands and a `physical` object containing the original camera
evaluator record. In that nested record, `applied_opd_nm` and
`final_applied_opd_nm` retain their physical meaning. Final-command camera
observation flags compare actual applied commands. Submission acquires no
additional exposure; final evaluator WFE is not a measured corrected image.

If a produced camera image fails public validation, `expose` raises
`NumericalObservationError` and the episode becomes terminal. Its consumed
exposure and photons remain charged; further exposure or submission is refused.
`evaluator_record()` then reports `numerical_failure`, both produced and public
observation counts, and the partial physical trajectory, with no final-submission
metrics. The private `failed_observation` retains an immutable detached copy of
the physical measurement, including nonfinite pixels. Writers must retain that
array separately using a format such as NPY; it is never a policy observation.

```python
from probebench.environment import EpisodeConfig
from probebench.nominal_calibration import GainEpisode
from probebench.optics import OpticalModel, OpticsConfig

model = OpticalModel(OpticsConfig())
instrument = EpisodeConfig(max_exposures=1, photon_budget=2_000,
                           photons_per_exposure=2_000)
# Driver/evaluator inputs: these must not be given to a policy.
episode = GainEpisode(model, instrument, [30, 12, -10, 8, -6, 15],
                      detector_seed=42, command_gain=0.9)
measurement = episode.expose([80, 0, 0, 0, 0, 0])
assert measurement.metadata()["nominal_applied_opd_nm"][0] == 80
evaluation = episode.submit([80, 0, 0, 0, 0, 0])
assert evaluation["physical"]["final_applied_opd_nm"][0] == 72
```

`run_classical_episode(...)` runs the declared exposure count using one existing
classical method and its supplied `PolicyConfig`. It returns public observation
objects, the private evaluation, complete policy settings and retained fit
diagnostics, including numerical fallbacks. It makes no parameter selection and
does not write artifacts, assign cohorts, manage seeds or bind source provenance.
On terminal camera failure it returns the partial evaluation and private
`failed_observation` without calling the policy's final-command method; on success
that field is `None`. A caller must inspect evaluation status before scoring.
The caller must establish those contracts before using it in a study. At `g=1`,
the same model, instrument, truth, seeds and settings reproduce the legacy
images, commands, physical evaluation and fit diagnostics; elapsed fit times
are not deterministic.

This is an API boundary between trusted local components, not isolation from
arbitrary Python code. Policies must not receive the driver, private evaluation
or underlying physical camera configuration. See [the execution trust
model](../SECURITY.md). The new schema is not accepted by existing agent,
archive-export or replay formats without an explicit adapter.

The separate [scalar-gain development runner](scalar-gain-study.md) supplies a
fixed study design, source-bound retention and verification around this adapter.
