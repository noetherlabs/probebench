# Noise-aware phase-diversity development

This is a development extension. The original least-squares baseline and retained
development run remain unchanged. No final cohort or superiority claim is defined here.

## Reason for the change

The camera draws a Poisson photoelectron count followed by additive Gaussian read noise.
An unweighted intensity fit need not use those measurements efficiently. Prior work
compares [Gaussian and Poisson phase-diversity objectives](https://arxiv.org/abs/2308.00734)
and discusses [mixed photon/read noise in computational imaging](https://arxiv.org/html/2308.02436v2).
Those papers motivate checking the noise model; their results do not establish performance
of this implementation. The latter paper uses a practical Gaussian approximation;
the implementation here evaluates the discrete Poisson-Gaussian convolution instead.

For expected photoelectrons mu, measured value y and known read-noise standard deviation s,

```text
p(y | mu,s) = sum(k=0..infinity) Poisson(k | mu) Normal(y-k | 0,s^2)
loss = -sum(pixels, exposures) log p
d(loss)/d(mu) = 1 - E[k | y,mu,s] / mu
```

Negative read-noise samples remain in the likelihood. For s=0, use the Poisson mass
directly and require nonnegative integer observations. Optical intensity derivatives
are analytic: differentiate the pupil phasor, propagate each derivative by Fourier
transform, and compute `2 Re(conj(field) * field_derivative)`.

## Truncation and numerical convention

For unnormalized summands w(k), the adjacent ratio is

```text
log(w(k+1)/w(k)) = log(mu) - log(k+1) + (y-k-1/2)/s^2.
```

It strictly decreases with k. Binary search locates a discrete mode. The search upper
bound `max(0, ceil(y + s^2 log(1+mu))) + 1` lies beyond the ratio's zero. Evaluate the
sum within H counts of the mode, clipped at k=0. Successive log ratios decrease by
at least `1/s^2`; therefore the omitted mass divided by the mode mass, and hence an
upper bound on omitted mass relative to the retained sum, is

```text
2 exp(-H(H+1)/(2s^2)) / (1 - exp(-(H+1)/s^2)).
```

Choose H until this bound is at most 1e-12. This is a mathematical truncation bound,
not a bound on floating-point rounding or derivative error. Check values against a
wide independent sum and gradients against finite differences.

Means below 1e-12 photoelectrons use an explicit numerical floor with zero derivative
inside the floor. This avoids singular derivatives at exact dark pixels. It is not
additional physical background or photons credited to the camera. Its effect must be
kept distinct from the finite-sum bound.

## Optimizer and interpretation

Use bounded L-BFGS-B with analytic gradients and the same three declared starting
defocus guesses as the initial least-squares fit. The installed SciPy interface was
checked against the opened [official minimize documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html).
Each fit retains all start statuses and costs, selects the lowest finite objective,
and reports whether that selected fit converged. A finite unconverged fit is not silently
reported as convergence. No finite fit raises an explicit failure.

## Acceptance checks

```text
analytic optical Jacobian --> directional differences + zero total-power derivative
pixel likelihood          --> wide independent Poisson/normal sum
pixel derivative          --> independent finite differences
phase objective           --> six-parameter directional differences
bounded optimizer         --> independent noiseless/bright synthetic checks
low-photon pilot           --> shared observed frames; retain every fit and failure
```

The optimizer can still reach a local minimum. Shared images are appropriate for this
estimator comparison because all estimators use the same fixed acquisition commands;
do not reuse identical frames for controllers whose actions alter the optical field.

## Planned development acquisition

`configs/noise_qualification_development.json` declares four six-mode wavefronts,
two detector realizations per wavefront, and 2,000 / 20,000 / 200,000 expected photons
per exposure. The planned acquisition uses zero command followed by +80 and -80 nm
defocus commands. Thus the planned dataset contains 24 measured cases and 48 estimator
fits. These counts are planned until a complete, verified run exists.

The four wavefronts comprise two conjugate-inversion pairs: negate even modes and keep
odd coma modes unchanged. This preserves the initial ideal intensity for the symmetric
pupil. Replicate seeds are explicitly enumerated and separated from the earlier demo.
Three exposures are charged in full. Read-noise level and all optical calibration are
given to both estimators; their acquisition/calibration cost is outside this idealized
task, not silently inferred from extra uncounted images.

Each case saves the acquired images before fitting. Resuming a case without a completed
fit reuses its saved acquisition. Completed cases are immutable and skipped only after
source/configuration/runtime, case identity, record digest and raw archive hash checks.
An exclusive process lock prevents concurrent writers. A changed binding refuses resume.
Partial final report publication can be completed from retained cases without refitting.
An operating reserve of 1 GiB plus a conservative raw-evidence envelope is required;
this has no effect on the separate frozen controller study's much larger storage gate.

The pilot reports every case and solver convergence status. It does not use correlated
image pixels as experimental samples and does not claim a population ranking from two
related wavefront pairs. Equal physical measurements do not imply equal optimization
effort; solver differences and failure counts remain explicit.
