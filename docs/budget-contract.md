# Multi-budget classical development contract

This is an implementation contract for known development fixtures, not a preregistration
of a final benchmark. Source and configuration must be frozen before each retained run.
Any final population, model selection or significance claim requires a separate reviewed
protocol and reserved cohort. The original three-exposure demonstration remains intact.

## Physical budget and submission

Every method gets the same instrument calibration, per-mode OPD limits, maximum exposure
count and expected incident photon allowance. Each acquired frame costs its full exposure
and photon amount, including probes and unsuccessful actions. Images retain signed read
noise. The public display transform is fixed. Truth, detector seeds and evaluator metrics
are not policy inputs.

A final correction may be inferred from previous measurements and submitted without a
new exposure. This applies to every method. The evaluator scores its optical field, and
the record explicitly states whether that exact final command was measured. A final
unmeasured correction must never be displayed as a measured star image. A separate
validation exposure, if requested, costs another exposure and its photons.

This differs from the initial demonstration's phase-diversity policy, which submitted
its last measured correction. Keep the policy version and this distinction visible;
do not combine results under the old method label.

## Development references

- **No action:** acquire the allotted frames at zero command and submit zero.
- **Random probe:** first acquire at zero; thereafter choose independent bounded six-mode
  commands from a fixed public distribution. Submit the measured command with greatest
  central 3×3 flux divided by its expected incident photons.
- **Coordinate search:** first acquire at zero. For each of the six modes, acquire +step
  and −step around the same incumbent command. After both observations, select the best
  measured candidate or retain the incumbent using the same camera-only score. Advance
  through all six modes. Halve the step after a complete sweep, down to a declared floor.
  A complete first sweep requires 13 exposures, including the initial image. A smaller
  budget is a partial search, not a six-mode solution. Clip commands to public limits
  before acquiring; even duplicate commands at a bound consume the full measurement cost.
- **Sequential phase diversity:** first acquire at zero, then at a known positive defocus
  offset. Use the validated noise-aware estimator on available frames. Subsequent
  acquisitions alternate correction measurements with known defocus probes about the
  inferred correction; probe amplitude and ordering are fixed in the configuration.
  Submit the estimate from all available frames. With only one observation, this
  reference abstains and submits zero because its diversity estimator requires distinct
  commands. Retain all fit starts and any finite nonconvergence or numerical failure.

The derivative-free score is an observed flux proxy; it is not the evaluator's noiseless
Strehl or residual WFE. It can select a noisy maximum. The estimator has an exactly matched
optical model in the first development comparison; that advantage is part of its declared
calibration, not evidence of robustness to model mismatch.

## Measurements, pairing and failure handling

Report residual pupil WFE, sampled peak Strehl, actual photons and exposures used, final
command measurement status, and peak temporary WFE increase. Preserve every trajectory.
Keep optimization work separate from physical exposure budgets; record solver calls and
statuses without converting them into a real-time performance claim.

Use matched hidden development wavefronts and declared detector streams across methods.
Do not give different actions identical camera frames: the plant intensity determines
each frame's count distribution. Budget settings may share random streams to support
paired curves, so they are correlated observations, not independent experimental samples.

All numerical failures count in the table. No post hoc replacement draw, seed deletion,
or tuning on a final reserved cohort is allowed. The initial curve run establishes that
the implementation and accounting work; its small known fixture set cannot rank general
method capability or justify population uncertainty estimates.

## Checks before retained acquisition

1. Both coordinate probes use the same anchor, all six modes are visited, and bounds and
   partial sweeps are respected without consulting hidden state.
2. Submissions use only the available observation prefix; unmeasured final commands are
   labelled accurately and acquire no uncharged camera data.
3. A supplied synthetic image history can drive policy decisions without any Episode or
   evaluator object. Replaying the same history and public random seed is deterministic.
4. Resource admission, immutable source/configuration binding, resume integrity and all
   failure records pass before the development curve is collected.
