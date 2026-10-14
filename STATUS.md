# Status

## 2026-10-14 — noise-aware development qualification complete

Destination: private `noetherlabs/probebench`. The initial foundation is on the repository's
default branch, `codex/probebench-foundation`. The current increment is on
`codex/probebench-noise-validation`; its acquisition source was frozen at commit `816bec5`.
The existing noether-ao checkout, its dirty files and its scheduler are untouched.

The native task goal and hourly continuation remain active. Three initial subagents
stopped with workspace-credit errors; none should be described as running. Continue
useful local work; do not retry failed agent dispatches or consume reset credits without
authorization. Local compute only remains the spending constraint.

## Verified evidence

- The independent pinned environment installed successfully. It passes **57 checks**,
  covering physical propagation, signed camera noise, accounting, observation separation,
  likelihood values/gradients, optimizer failures and interrupted-run integrity.
- The original 16-episode development run reproduced in that fresh environment. All
  **59 retained files match exactly**, including raw observations and evaluator results.
  [Verification record](results/fresh-environment-reproduction-20261014.json).
- The [noise qualification](results/noise-qualification-20261014/report.md) retains
  **24 acquired cases, 48 selected estimator fits and 144 optimizer starts**. All selected
  fits and starts converged. Least squares and the mixed Poisson–Gaussian estimator used
  identical measured frames in each case. The resumed run preserved the first chunk's
  eight acquisition/record files exactly; manifest and case verification pass.
- At 6,000 / 60,000 / 600,000 total expected photons, mean residual WFE was
  **4.2786 / 1.3128 / 0.4257 nm** for least squares and **2.3765 / 0.7720 / 0.2146 nm**
  for the noise-aware estimator. The latter improved 22 of 24 individual cases; both
  exceptions are retained. The [all-case figure](results/noise-analysis-20261014/all-fits.png)
  was visually inspected. Its plot source and input/output hashes are recorded.

These are known development fixtures with an exact plant/twin match. There is no
population ranking, significance claim, held-out benchmark or telescope-performance
claim. The qualification evaluates inferred final corrections without taking a final
corrected exposure. It compares different solvers as well as different likelihoods.

## Remaining scope and constraints

The [multi-budget development contract](docs/budget-contract.md) specifies a genuine
six-mode coordinate search and explicit final-submission accounting. Implementation and
retained budget curves are next. The original three-exposure coordinate search reaches
only defocus and must not be represented as a complete six-mode baseline.

No multimodal agent comparison, public release, full controller study, fault-recovery
study or twin-qualification study is complete. In-process observation separation remains
an API boundary, not hostile-code isolation. Expert optics review and a separate frozen
evaluation protocol are still required before final ProbeBench claims.

Full controller acquisition remains subject to its unchanged **105,033,365,984-byte**
storage gate. Current local capacity is insufficient; a question about another local
volume is pending. Never lower that gate or delete unrelated files to satisfy it.
Website redesign is assigned to Claude. Public release identity/licensing remain open.

Next action: implement and verify the declared six-mode/budget policies, then retain a
small development curve before building the isolated agent interface and replay explorer.
