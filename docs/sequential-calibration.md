# Sequential calibration with bounded measurements

The sequential experiment compares four classical policies on the same static,
six-mode optical plant. The independent resource is the number of acquired exposures.
At a fixed expected photon count per exposure, increasing the exposure allowance also
increases the photon allowance. These curves do not isolate exposure count from total
photon count.

## Observation and submission

Every exposure contains a signed Poisson–Gaussian image and the applied OPD command.
Policies receive the public optical model, detector calibration, command bounds and
observation history. They do not receive latent phase, detector seeds or evaluator
quality metrics. The random policy uses a separate, public random stream.

All four methods acquire exactly the allowed number of exposures. A final command can
be submitted without acquiring another image. The evaluator computes residual WFE and
sampled peak Strehl at that command. This evaluation is not a camera measurement and
does not incur photons. Any experimental validation of the correction would require
another charged exposure.

The evaluator records every earlier exposure whose command exactly equals the final
command, and separately records whether it equals the last exposure. An unmeasured
final correction must not be illustrated as a measured corrected image.

## Policies

All methods start with a zero-command exposure. For derivative-free selection, the
camera score is the signed electron sum in the central 3×3 detector pixels divided by
the expected incident photons. This noisy flux proxy is distinct from evaluator Strehl.
Ties retain the first candidate in acquisition order.

| Identifier | Acquisition and submission rule |
| --- | --- |
| `no_action_v2` | Acquire at zero throughout and submit zero. |
| `random_probe_v2` | Draw each subsequent six-mode command independently and uniformly within ±60 nm OPD per mode, clipped to instrument bounds. Submit the measured command with the greatest camera score. |
| `coordinate_search_v2` | Acquire ±40 nm about the same incumbent in each mode, select the highest-scoring candidate after the pair, then advance through the six modes. Halve the step after each complete sweep, down to 5 nm. Submit the best measured command, including a completed positive probe from a partial pair. |
| `phase_diversity_mle_v1` | Acquire a known +80 nm defocus command after the initial frame. Infer the wavefront using all available images. Alternate inferred-correction exposures with defocus probes about the current estimate, starting at −80 nm and alternating sign. Submit the estimate using all acquired images; with one image, submit zero. |

The numeric settings are specified by the supplied configuration. Clipped or repeated
commands still consume a complete exposure and its photons. Coordinate search requires
13 exposures for its first full six-mode sweep, and 25 for its second. Smaller budgets
are partial searches. Policy decisions depend on the available history, not on the
total budget; paired budget settings therefore share acquisition prefixes.

Phase estimation uses the [Poisson–Gaussian likelihood](noise-likelihood.md), analytic
optical derivatives and bounded multistart L-BFGS-B. All starts and selected-fit statuses
are retained. A finite nonconverged optimum remains an explicitly flagged estimate.
A numerical failure submits or probes about the best measured camera-score command
and remains counted in the results. No replacement wavefront or noise draw is used.

The estimator assumes an exactly matched optical model. Its comparison with flux-based
search consequently combines differences in inference, calibration and measurement
selection. It does not isolate any one of those factors.

The separate [measurement-schedule ablation](measurement-schedule-ablation.md) adds
`unprobed_mle_v1` and `fixed_diversity_mle_v1` to compare acquisition schedules with a
common estimator. These controls are not part of the original four-policy dataset.

## Development design

`configs/sequential_calibration_development.json` specifies four known mixed-mode
wavefronts forming two conjugate-ambiguity pairs, two detector replicates and exposure
allowances of 1, 2, 3, 7, 13 and 25. Each exposure has 20,000 expected incident photons
and 1-electron read noise. The full design contains 192 episodes and 1,632 exposures.
These are design counts, not a statement that a particular run completed.

Methods and budget settings share declared wavefront/noise stream identities. Different
commands produce different detector distributions; paired seeds do not imply identical
images after policies diverge. Budget points are correlated and must not be treated as
independent experimental samples. The four public wavefronts are development fixtures,
not a reserved population for model selection or generalization claims.

## Execution

```sh
uv run --locked python -m probebench.sequential \
  --config configs/sequential_calibration_development.json \
  --output work/sequential-calibration
uv run --locked python -m probebench.sequential \
  --verify work/sequential-calibration
```

The runner checks disk capacity, acquires an exclusive directory lock and snapshots the
source, configuration and numerical runtime. Each exposure is published atomically
before the next policy decision. Compressed records include the image, public metadata,
completed fit records and content hashes. Completed episodes are published separately;
the full manifest is written only after every episode and derived table is complete.

For an interrupted run, repeat the command with `--resume`. Source, configuration and
runtime must match. Saved exposures restore the simulator's random state by deterministic
replay and are compared exactly with their retained images. Replay does not replace the
retained evidence or supply extra observations to the policy. Saved estimator fits are
reused; an interrupted, unpublished fit can be recomputed. Completed episodes are checked
without executing their policies again. `--limit-episodes N` permits deliberate chunks
of N new episodes, leaving the output explicitly incomplete until final publication.

Verification checks file hashes, case/source bindings, resource accounting, checkpoint
consistency and evaluator metrics recomputed from latent phase and recorded commands.
It does not independently repeat each optimization or authenticate the archive's author.
See the [artifact trust model](../SECURITY.md) before processing an unfamiliar archive.

Decision and fit durations are elapsed local measurements subject to machine load and
scheduling. They are not real-time control guarantees.
