# Sequential optical calibration under exposure budgets

Known development wavefronts; static, monochromatic, exactly matched plant and estimator.
Final corrections may be submitted without another exposure. Evaluator WFE and Strehl
for an unmeasured correction must not be presented as a measured corrected image.

| Photons/exposure | Exposures | Method | Episodes | Mean WFE (nm) | Mean Strehl | Mean peak probe increase (nm) | Episodes with failed fits | Episodes with unconverged fits |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2000 | 1 | unprobed_mle_v1 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 2000 | 1 | fixed_diversity_mle_v1 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 2000 | 1 | phase_diversity_mle_v1 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 2000 | 2 | unprobed_mle_v1 | 8 | 36.2662 | 0.772933 | 0.0000 | 0 | 0 |
| 2000 | 2 | fixed_diversity_mle_v1 | 8 | 2.8453 | 0.998810 | 47.0823 | 0 | 0 |
| 2000 | 2 | phase_diversity_mle_v1 | 8 | 2.8453 | 0.998810 | 47.0823 | 0 | 0 |
| 2000 | 3 | unprobed_mle_v1 | 8 | 26.8787 | 0.836199 | 0.0000 | 0 | 0 |
| 2000 | 3 | fixed_diversity_mle_v1 | 8 | 2.1445 | 0.999341 | 73.6926 | 0 | 0 |
| 2000 | 3 | phase_diversity_mle_v1 | 8 | 2.5390 | 0.999036 | 47.0823 | 0 | 0 |
| 2000 | 7 | unprobed_mle_v1 | 8 | 35.8526 | 0.772999 | 0.0000 | 0 | 0 |
| 2000 | 7 | fixed_diversity_mle_v1 | 8 | 1.4428 | 0.999711 | 73.6926 | 0 | 0 |
| 2000 | 7 | phase_diversity_mle_v1 | 8 | 2.1438 | 0.999337 | 58.8240 | 0 | 0 |
| 20000 | 1 | unprobed_mle_v1 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 20000 | 1 | fixed_diversity_mle_v1 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 20000 | 1 | phase_diversity_mle_v1 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 20000 | 2 | unprobed_mle_v1 | 8 | 27.4425 | 0.823427 | 0.0000 | 0 | 0 |
| 20000 | 2 | fixed_diversity_mle_v1 | 8 | 0.8430 | 0.999895 | 47.0823 | 0 | 0 |
| 20000 | 2 | phase_diversity_mle_v1 | 8 | 0.8430 | 0.999895 | 47.0823 | 0 | 0 |
| 20000 | 3 | unprobed_mle_v1 | 8 | 35.4143 | 0.771960 | 0.0000 | 0 | 0 |
| 20000 | 3 | fixed_diversity_mle_v1 | 8 | 0.7349 | 0.999925 | 73.6926 | 0 | 0 |
| 20000 | 3 | phase_diversity_mle_v1 | 8 | 0.7746 | 0.999909 | 47.0823 | 0 | 0 |
| 20000 | 7 | unprobed_mle_v1 | 8 | 36.3887 | 0.762475 | 0.0000 | 0 | 0 |
| 20000 | 7 | fixed_diversity_mle_v1 | 8 | 0.4506 | 0.999971 | 73.6926 | 0 | 0 |
| 20000 | 7 | phase_diversity_mle_v1 | 8 | 0.4713 | 0.999969 | 58.6073 | 0 | 0 |

Every episode is included. A numerical fit failure uses the best measured camera-score
command as a declared fallback and remains counted above. Finite nonconvergence remains
visible. Small related fixture sets and shared noise streams do not support population
rankings, significance claims or treating budget points as independent replicates.

Coordinate search completes its first six-mode sweep at 13 exposures. The phase-diversity
reference submits zero after one exposure, then uses known probes and all available frames.
Reported decision durations are recorded local execution costs, not real-time guarantees.

The schedule controls retain the same estimator. Unprobed estimation acquires
only zero-command images; its selected minimum is not an identified wavefront.
Fixed diversity repeats 0,+d,-d defocus commands without using camera feedback
during acquisition. Both fit all available images once at final submission;
all three estimator policies submit zero with only one exposure.
