# Engineering review — first portable slice

## What already exists

The existing private noether-ao checkout contains a scalar Fourier propagator, JAX/dLux
plant, delayed controls, pyramid sensor, immutable study ledger, frozen selection and
storage admission. Reuse those contracts after repository integration. Their historical
results are not new evidence from this task. Existing dirty observability work is excluded
from edits. This prototype provides a small independent scalar reference and a new
low-order focal-camera task, not a replacement implementation of delayed AO control.

## Scope challenge and recommendation

Retain all four studies in the programme, but implement the smallest useful common
boundary first. A general agent platform, queue service or cloud scheduler would add
unnecessary state: native task goals and small source-bound checkpoints suffice now.
The separate optical/environment/policy/run boundaries are justified by information
isolation and independent checks; do not create a class hierarchy for study variants.
The user's chosen release order is ProbeBench with the controller study, then recovery
and twin qualification. No architecture decision requires delaying this reversible slice.

## Architecture and test diagram

```text
public instrument calibration + observation history
                     |
                policy action -------- invalid / over budget --> reject, no mutation [test]
                     |
                applied OPD [unit/sign test]
                     |
private true OPD --> residual pupil --> Fourier field [matrix-DFT/energy test]
                     |                       |
          private evaluator           Poisson + read noise [seed/accounting test]
          WFE, Strehl, penalty                |
                     |                public observation [no truth/seed fields test]
                     |                       |
              evaluator files         observation files
                     +-----------+-----------+
                          immutable manifest [hash/tamper/overwrite tests]
                                 |
                         replay / release package (later)
```

## Main risks and choices

1. Sampled modes are not perfectly orthogonal. Use pupil-array residual RMS and retain
   the basis normalization. Test a mixed wavefront instead of assuming coefficient norm.
2. The in-process policy API is not hostile-code isolation. Pass observations only and
   explicitly defer external agent execution until a separate transport prevents access.
3. Noisy least squares may be statistically inefficient. Verify noiseless recovery and
   retain noisy outcomes; qualify the likelihood and convergence before final evaluation.
4. Probe exposures can harm the image. Count every exposure and record private quality
   along the whole trajectory. Show intermediate degradation, not only the final result.
5. Disk interruption could leave incomplete evidence. Create a unique run directory,
   preserve incomplete output and publish the final manifest last. Refuse overwrite;
   verification must reject a missing manifest or a changed artifact.

No silent failure is accepted at these boundaries. Tests cover explicit rejection;
the command exits with an error rather than manufacturing a successful run.

## NOT in scope for this slice

- Website redesign: assigned to Claude by the user.
- Reserved controller-study acquisition: original storage/source gates not satisfied.
- Paid model calls, cloud compute, sending email: not authorized.
- Public uploads/licence changes: repository and release identity unresolved.
- A four-study final leaderboard: requires validated protocols and complete cohorts.

## Review completion

Scope challenge and existing-work inventory complete; architecture/data-flow review
complete for the development slice; test diagram above; 32 correctness tests and a
16-episode retained development demonstration pass. A gstack checklist review of the
initial source changes found no additional blocking issue within this explicitly
development-only scope; there was no pre-existing Git base branch. Primary gaps before
final studies are isolated agent execution, statistical
estimator qualification, expert optics review, reserved protocol and storage capacity.
