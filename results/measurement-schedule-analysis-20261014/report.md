# Measurement-schedule development results

192 episodes and 624 charged exposures. Every configured case is included. Final scores are evaluator quantities; final commands may be unmeasured.

| Photons/exposure | Exposures | Method | Mean WFE (nm) | Mean WFE modulo conjugacy (nm) | Mean Strehl | Mean peak WFE increase (nm) | Failed / nonconverged selected fits |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2000 | 1 | unprobed_mle_v1 | 36.8782 | 36.8782 | 0.839822 | 0.0000 | 0 / 0 |
| 2000 | 1 | fixed_diversity_mle_v1 | 36.8782 | 36.8782 | 0.839822 | 0.0000 | 0 / 0 |
| 2000 | 1 | phase_diversity_mle_v1 | 36.8782 | 36.8782 | 0.839822 | 0.0000 | 0 / 0 |
| 2000 | 2 | unprobed_mle_v1 | 36.2662 | 2.7988 | 0.772933 | 0.0000 | 0 / 0 |
| 2000 | 2 | fixed_diversity_mle_v1 | 2.8453 | 2.8453 | 0.998810 | 47.0823 | 0 / 0 |
| 2000 | 2 | phase_diversity_mle_v1 | 2.8453 | 2.8453 | 0.998810 | 47.0823 | 0 / 0 |
| 2000 | 3 | unprobed_mle_v1 | 26.8787 | 2.2636 | 0.836199 | 0.0000 | 0 / 0 |
| 2000 | 3 | fixed_diversity_mle_v1 | 2.1445 | 2.1445 | 0.999341 | 73.6926 | 0 / 0 |
| 2000 | 3 | phase_diversity_mle_v1 | 2.5390 | 2.5390 | 0.999036 | 47.0823 | 0 / 0 |
| 2000 | 7 | unprobed_mle_v1 | 35.8526 | 1.7308 | 0.772999 | 0.0000 | 0 / 0 |
| 2000 | 7 | fixed_diversity_mle_v1 | 1.4428 | 1.4428 | 0.999711 | 73.6926 | 0 / 0 |
| 2000 | 7 | phase_diversity_mle_v1 | 2.1438 | 2.1438 | 0.999337 | 58.8240 | 0 / 0 |
| 20000 | 1 | unprobed_mle_v1 | 36.8782 | 36.8782 | 0.839822 | 0.0000 | 0 / 0 |
| 20000 | 1 | fixed_diversity_mle_v1 | 36.8782 | 36.8782 | 0.839822 | 0.0000 | 0 / 0 |
| 20000 | 1 | phase_diversity_mle_v1 | 36.8782 | 36.8782 | 0.839822 | 0.0000 | 0 / 0 |
| 20000 | 2 | unprobed_mle_v1 | 27.4425 | 0.9030 | 0.823427 | 0.0000 | 0 / 0 |
| 20000 | 2 | fixed_diversity_mle_v1 | 0.8430 | 0.8430 | 0.999895 | 47.0823 | 0 / 0 |
| 20000 | 2 | phase_diversity_mle_v1 | 0.8430 | 0.8430 | 0.999895 | 47.0823 | 0 / 0 |
| 20000 | 3 | unprobed_mle_v1 | 35.4143 | 0.8119 | 0.771960 | 0.0000 | 0 / 0 |
| 20000 | 3 | fixed_diversity_mle_v1 | 0.7349 | 0.7349 | 0.999925 | 73.6926 | 0 / 0 |
| 20000 | 3 | phase_diversity_mle_v1 | 0.7746 | 0.7746 | 0.999909 | 47.0823 | 0 / 0 |
| 20000 | 7 | unprobed_mle_v1 | 36.3887 | 0.5484 | 0.762475 | 0.0000 | 0 / 0 |
| 20000 | 7 | fixed_diversity_mle_v1 | 0.4506 | 0.4506 | 0.999971 | 73.6926 | 0 / 0 |
| 20000 | 7 | phase_diversity_mle_v1 | 0.4713 | 0.4713 | 0.999969 | 58.6073 | 0 / 0 |

## Fixed versus adaptive diversity

Differences are fixed minus adaptive WFE. Positive values favour adaptive acquisition.

| Photons/exposure | Exposures | Mean difference (nm) | Adaptive lower | Fixed lower | Exact ties |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 1 | 0.000000 | 0 | 0 | 8 |
| 2000 | 2 | 0.000000 | 0 | 0 | 8 |
| 2000 | 3 | -0.394440 | 2 | 6 | 0 |
| 2000 | 7 | -0.701034 | 0 | 8 | 0 |
| 20000 | 1 | 0.000000 | 0 | 0 | 8 |
| 20000 | 2 | 0.000000 | 0 | 0 | 8 |
| 20000 | 3 | -0.039694 | 3 | 5 | 0 |
| 20000 | 7 | -0.020638 | 4 | 4 | 0 |

240 selected fits and 720 starts were retained. 0 selected fits failed numerically; 0 selected fits and 0 individual starts did not converge.

32 paired fixed/adaptive cases at one or two exposures have identical images and submitted commands. The largest unprobed conjugate-likelihood difference is 8.882e-16 per expected photon.

All single-exposure methods abstain and submit zero. Modulo-conjugacy error is an evaluator-only diagnostic, not the correction metric. A small value can coexist with a wrong physical correction.

The four wavefront fixtures and shared noise streams are correlated. These descriptive results do not support population significance claims, a generally preferred schedule, or conclusions about model mismatch, multimodal agents or telescope performance. Numerical convergence does not establish identifiability. All paired differences, per-episode metrics and checks are retained in analysis.json.
