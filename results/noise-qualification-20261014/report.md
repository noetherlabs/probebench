# Noise-aware phase diversity: development qualification

This comparison uses identical retained exposures for both estimators. It evaluates
estimated corrections without taking a final corrected camera image; it is not a
closed-loop controller benchmark. Fixtures are public development cases only.

| Expected photons used | Estimator | Finite fits | Converged fits | Mean residual WFE (nm) |
| ---: | --- | ---: | ---: | ---: |
| 6000 | least_squares | 8/8 | 8/8 | 4.2786 |
| 6000 | poisson_gaussian_mle | 8/8 | 8/8 | 2.3765 |
| 60000 | least_squares | 8/8 | 8/8 | 1.3128 |
| 60000 | poisson_gaussian_mle | 8/8 | 8/8 | 0.7720 |
| 600000 | least_squares | 8/8 | 8/8 | 0.4257 |
| 600000 | poisson_gaussian_mle | 8/8 | 8/8 | 0.2146 |

Means describe finite fits; the adjacent failure counts must accompany them.
All individual fits, all optimizer starts, measured images and known calibration
commands are retained. Four related wavefronts and two noise draws per setting do
not justify a population ranking or a significance claim. This pilot uses an exact
plant/twin match; robustness to optical mismatch remains untested. The two solvers
have different optimization procedures, so this does not isolate likelihood choice
from every convergence effect. Read the complete per-start records.
