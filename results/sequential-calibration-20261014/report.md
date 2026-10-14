# Sequential optical calibration under exposure budgets

Known development wavefronts; static, monochromatic, exactly matched plant and estimator.
Final corrections may be submitted without another exposure. Evaluator WFE and Strehl
for an unmeasured correction must not be presented as a measured corrected image.

| Photons/exposure | Exposures | Method | Episodes | Mean WFE (nm) | Mean Strehl | Mean peak probe increase (nm) | Episodes with failed fits | Episodes with unconverged fits |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20000 | 1 | no_action_v2 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 20000 | 1 | random_probe_v2 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 20000 | 1 | coordinate_search_v2 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 20000 | 1 | phase_diversity_mle_v1 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 20000 | 2 | no_action_v2 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 20000 | 2 | random_probe_v2 | 8 | 36.8782 | 0.839822 | 66.7989 | 0 | 0 |
| 20000 | 2 | coordinate_search_v2 | 8 | 31.9239 | 0.874321 | 17.5890 | 0 | 0 |
| 20000 | 2 | phase_diversity_mle_v1 | 8 | 0.7131 | 0.999929 | 47.0823 | 0 | 0 |
| 20000 | 3 | no_action_v2 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 20000 | 3 | random_probe_v2 | 8 | 36.8782 | 0.839822 | 80.0940 | 0 | 0 |
| 20000 | 3 | coordinate_search_v2 | 8 | 26.9696 | 0.908819 | 35.1781 | 0 | 0 |
| 20000 | 3 | phase_diversity_mle_v1 | 8 | 0.6381 | 0.999944 | 47.0823 | 0 | 0 |
| 20000 | 7 | no_action_v2 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 20000 | 7 | random_probe_v2 | 8 | 36.8782 | 0.839822 | 86.5113 | 0 | 0 |
| 20000 | 7 | coordinate_search_v2 | 8 | 26.9696 | 0.908819 | 35.1781 | 0 | 0 |
| 20000 | 7 | phase_diversity_mle_v1 | 8 | 0.4204 | 0.999976 | 58.5215 | 0 | 0 |
| 20000 | 13 | no_action_v2 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 20000 | 13 | random_probe_v2 | 8 | 36.8782 | 0.839822 | 90.1462 | 0 | 0 |
| 20000 | 13 | coordinate_search_v2 | 8 | 26.9696 | 0.908819 | 35.1781 | 0 | 0 |
| 20000 | 13 | phase_diversity_mle_v1 | 8 | 0.3629 | 0.999981 | 58.5783 | 0 | 0 |
| 20000 | 25 | no_action_v2 | 8 | 36.8782 | 0.839822 | 0.0000 | 0 | 0 |
| 20000 | 25 | random_probe_v2 | 8 | 36.8782 | 0.839822 | 97.1055 | 0 | 0 |
| 20000 | 25 | coordinate_search_v2 | 8 | 20.8943 | 0.944401 | 35.1781 | 0 | 0 |
| 20000 | 25 | phase_diversity_mle_v1 | 8 | 0.2736 | 0.999990 | 58.5810 | 0 | 0 |

Every episode is included. A numerical fit failure uses the best measured camera-score
command as a declared fallback and remains counted above. Finite nonconvergence remains
visible. Small related fixture sets and shared noise streams do not support population
rankings, significance claims or treating budget points as independent replicates.

Coordinate search completes its first six-mode sweep at 13 exposures. The phase-diversity
reference submits zero after one exposure, then uses known probes and all available frames.
Reported decision durations are recorded local execution costs, not real-time guarantees.
