# Two indistinguishable star images, two opposite corrections

**Development demonstration, not final benchmark evidence.**

For a symmetric pupil, positive and negative pure defocus have the same
noiseless focal intensity. Noisy exposures therefore have the same distribution,
although independent noise draws need not produce identical pixel arrays.

A known positive OPD correction changes the two residual wavefronts differently.
Here the phase-diversity policy uses an initial exposure, a known probe, then
a third exposure to measure its inferred correction. The hidden wavefront never
enters the policy. All measurements use the same fixed display transform.

![Retained measurements](ambiguity.png)

## Complete development outcomes

Each row uses three exposures of 200,000 expected incident photons each, with
Poisson photoelectrons and 1-electron Gaussian read noise. The table is generated
from retained episode records. All four fixtures and all four policies are shown.

| Fixture | Policy | Initial WFE (nm) | Final WFE (nm) | Sampled Strehl | Peak probe WFE increase (nm) |
| --- | --- | ---: | ---: | ---: | ---: |
| positive_defocus | no_action | 45.000 | 45.000 | 0.762108 | 0.000 |
| positive_defocus | random_probe | 45.000 | 45.000 | 0.762108 | 41.758 |
| positive_defocus | coordinate_search | 45.000 | 5.000 | 0.996742 | 0.000 |
| positive_defocus | phase_diversity | 45.000 | 0.285 | 0.999989 | 0.000 |
| negative_defocus | no_action | 45.000 | 45.000 | 0.762108 | 0.000 |
| negative_defocus | random_probe | 45.000 | 45.000 | 0.762108 | 75.361 |
| negative_defocus | coordinate_search | 45.000 | 5.000 | 0.996742 | 40.000 |
| negative_defocus | phase_diversity | 45.000 | 0.427 | 0.999976 | 80.000 |
| mixed_a | no_action | 38.699 | 38.699 | 0.825011 | 0.000 |
| mixed_a | random_probe | 38.699 | 38.699 | 0.825011 | 21.779 |
| mixed_a | coordinate_search | 38.699 | 25.680 | 0.917209 | 0.000 |
| mixed_a | phase_diversity | 38.699 | 0.614 | 0.999951 | 16.268 |
| mixed_b | no_action | 35.058 | 35.058 | 0.854633 | 0.000 |
| mixed_b | random_probe | 35.058 | 35.058 | 0.854633 | 93.808 |
| mixed_b | coordinate_search | 35.058 | 28.259 | 0.900430 | 34.653 |
| mixed_b | phase_diversity | 35.058 | 0.311 | 0.999987 | 73.063 |

## What this does and does not establish

The development fixtures exercise information ambiguity, diagnostic intervention,
noise accounting and retained evidence. The probe temporarily worsens the negative
defocus case; its degradation is retained rather than hidden by a final-only plot.

Four known fixtures do not establish statistical superiority or generalization.
The phase-diversity reference uses multistart calibrated-intensity least squares;
its noise likelihood and convergence still need qualification. A three-exposure
coordinate search has only visited the defocus direction, so this is not a fair
full six-mode optimizer comparison. No multimodal agent was run. The sampled scalar
model is not telescope validation. Public development fixtures must never become
reserved final evaluation under a different label.

See configuration.json for fixtures, mode order, wavelength, limits and budgets;
observations/ for controller-visible records; evaluator/ for private truth and
per-exposure quality; source/ for the executed source snapshot and lockfile; and
manifest.json for content hashes and runtime versions.

Prior work: [pupil-diversity phase ambiguity](https://arxiv.org/abs/1607.01776) and
[active perception by optical agents](https://arxiv.org/html/2609.32918v1).
These sources motivate the task; this demonstration does not claim the first use
of optical intervention or AI in laboratory optics.
