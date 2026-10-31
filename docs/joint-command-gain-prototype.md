# Joint phase and command-response estimation: development prototype

This draft adds an offline estimator, not a runtime controller or a new claimed scientific method. The response parameter is a positive scalar multiplying commanded OPD. It is not the optical gain of a pyramid wavefront sensor.

For nominal command u_i, the forward model is I(a - g u_i). The existing Poisson-Gaussian camera likelihood is reused while fitting six phase coefficients a and one scalar response g. The final nominal correction is clipped(a_hat / g_hat). True phase and physical response are never estimator inputs.

`NominalObservation` copies and freezes camera arrays and nominal command values. Its legacy `applied_opd_nm` alias is explicitly nominal: it exists only to feed the unchanged baseline estimator without leaking physical response labels. This is not a claim that a requested command was physically applied.

## Executed checks

The added test file passed 16 tests in a local focused environment using byte-verified copies of the existing optical model, likelihood and nominal estimator at commit `9406f5b5b04f9692cd194599277109423a2dbee6`. Checks include all seven gradient coordinates, reduction to the nominal objective at unit gain, input immutability, refusal of unchanging commands, compensation algebra and a bright synthetic recovery fixture.

```sh
PYTHONPATH=src python -m pytest tests/test_gain_estimation.py -q
```

The entire repository suite, supported-version matrix, CI and hardware execution were not run in that local environment. Passing these tests does not qualify the prototype for deployment.

## Development evidence and promotion boundary

A separate retained local experiment compares unchanged nominal fixed and adaptive phase-diversity estimators with this joint estimator. It is not integrated into the repository's release or reserved evaluation cohorts. No existing controller, policy, result, frozen data or default behavior is changed by this draft.

Before promoting scientific claims in the repository, port and review the complete acquisition harness and raw artifacts, independently reproduce the comparison, and admit a new disjoint cohort. The local study uses a static six-mode, matched pupil and detector model, known photon flux, constant scalar gain and offline fitting. It does not establish performance under unknown flux, unmodeled optical aberrations, temporal gain variation, misregistration, mode-dependent response or real instrument measurements.

## Known limitations

Command diversity and bounded gain do not guarantee global identifiability. Three local starts are not proof of global optimality. The finite best fit is returned even when its optimizer reports failure, matching the old estimator's convention; every caller must inspect diagnostics. No confidence-calibrated withholding rule or safe command-authority gate is provided. The nominal model may perform better when calibration is already correct because the joint model estimates an additional parameter.

Phase diversity and deformable-mirror calibration have substantial prior art. This is a targeted engineering extension to ProbeBench, not a novelty or learned-control claim.
