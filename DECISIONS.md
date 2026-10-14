# Decisions

## 2026-10-14 — repository supplied

The user supplied `https://github.com/noetherlabs/probebench`. GitHub access verified it
was private and empty. Development moved into its own checkout on
`codex/probebench-foundation`; the earlier staging directory is superseded. No existing
scientific source or reserved protocol was transferred. The earlier decision to wait
for a destination is resolved. Public licence and publication choices remain unresolved.

## 2026-10-14 — scope and authority

The latest user request excludes website redesign, explicitly permits subagents and
requests sustained goal-based work. The user selected the recommended release order,
local work only, and a different repository to be supplied. The attached research brief
is a proposed programme, not evidence that experiments ran or permission to send messages.
Keep the existing noether-ao checkout, its dirty files, scheduler and reserved data intact.

## 2026-10-14 — portable first slice

Develop a small standalone optical reference and observation contract in this staging
directory. Do not assume the destination repository or establish a second competing AO
stack. The existing JAX/dLux plant is a candidate integration target once repository
access and protocol ownership are clear. This NumPy reference is useful for independent
checking and for a small CPU-only demonstration.

All displacement coefficients and commands are optical path difference in nanometres.
At normal incidence, an ideal reflective mirror surface displacement is half the applied
OPD. Positive correction is subtracted from the incoming wavefront. The six analytic
Zernike shapes are piston-removed and individually RMS-normalized on the finite pupil;
their finite-grid Gram matrix need not be exactly the identity. Compute WFE on the
actual pupil, never from coefficient norm alone.

The detector receives unit-power discrete focal intensity times the requested expected
photon count (unit throughput and quantum efficiency). Draw independent Poisson counts
and additive Gaussian read noise. Retain negative read-noise values. Photon budget means
expected incident photons, distinct from the realized detected photoelectron sum.
Keep a fixed public display transform; do not normalize each image by its peak.

Initial v0 fixtures, grids, stroke limits, probe amplitudes and optimization starts are
development choices, not a frozen final benchmark. Least-squares phase diversity is an
initial calibrated-intensity baseline, not yet a validated Poisson-Gaussian likelihood
estimator. No release claim may hide these distinctions.

## 2026-10-14 — model routing

Recommend Astra with high reasoning for experiment contracts, scientific ambiguities,
interpretation and cross-system decisions; Sol for settled implementation and tests;
Luna only for mechanical edits with deterministic checks. Existing agents inherit their
parent settings. No model switch, spending limit or reasoning level is claimed to be
enforced merely because a prompt recommends it.
