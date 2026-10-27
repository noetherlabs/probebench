# Optical model and measurement contract

## Propagation

The pupil is an interpixel-centred circular aperture sampled on a square grid. Six real
Zernike shapes describe defocus, cosine/sine astigmatism, horizontal/vertical coma and
spherical aberration. Each sampled shape is piston-subtracted and normalized to unit
RMS within the pupil. Finite sampling does not guarantee mutual orthogonality.

Let `a` be the latent OPD coefficients and `u` the applied correction, both in nanometres.
The residual wavefront is `B @ (a - u)`. The pupil phasor is normalized to unit discrete
power and propagated by a centred, orthonormal two-dimensional FFT. Zero padding changes
focal sampling without changing total power. Images are not renormalized independently.

The default configuration uses 32 pupil pixels, a padding factor of two and a wavelength
of 550 nm. These parameters define a numerical experiment; no physical telescope pupil,
actuator influence function or detector angular scale is implied.

## Detector

For each pixel, draw a Poisson variate with mean equal to the requested expected incident
photon count multiplied by normalized focal intensity. Add independent zero-mean Gaussian
read noise with the declared standard deviation, in electrons. The model assumes unit
throughput and quantum efficiency. Negative read-noise samples remain in the raw data.

Expected incident photons and realized photoelectrons are distinct quantities. Exposure
and photon limits are validated before advancing detector randomness or changing the
episode state. Invalid commands, nonfinite coefficients and exhausted budgets fail
explicitly.

The displayed image uses `log1p(max(image, 0)) / log1p(expected_photons)` with fixed colour
limits. This transformation is for display only; estimators receive signed raw values.
The image maximum is never used to normalize display brightness.

## Information boundary

An observation contains its index, applied OPD command, expected photons, cumulative
expected photons and measured image. The controller also receives declared optical
calibration and instrument limits. Latent coefficients, detector seeds, noiseless
intensity, residual WFE and Strehl belong to the evaluator.

This is an API-level separation in a trusted local simulation. It does not constrain
arbitrary Python code executing in the same process or with access to evaluator files.

The opt-in [nominal calibration adapter](nominal-calibration.md) separates requested
commands from actual response when simulating a constant scalar command-gain error.
Its versioned observation contract leaves the original known-OPD schema unchanged.

## Evaluation

- **Residual WFE:** piston-removed RMS of the sampled residual wavefront over the pupil.
  It is not approximated by the Euclidean norm of the mode coefficients.
- **Sampled peak Strehl:** maximum residual focal intensity divided by the maximum
  unaberrated focal intensity on the same detector grid.
- **Measurement resources:** acquired exposure count and total expected incident photons.
- **Probe degradation:** maximum measured-step WFE above initial WFE, clipped below at zero.

An ideal reflective surface displacement at normal incidence is half the corresponding
OPD. Positive correction is subtracted from the incident wavefront.

## Numerical validation

Tests compare the FFT propagator with an explicit matrix Fourier sum, check unit power,
piston removal and sampled-pupil RMS, and verify defocus sign ambiguity and its resolution
under known diversity. Optical and likelihood derivatives are checked against independent
finite differences. The Poisson–Gaussian likelihood is compared with a broad sum formed
from independent distribution functions.
