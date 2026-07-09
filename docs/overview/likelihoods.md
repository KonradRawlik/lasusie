# Built-in likelihoods

All likelihoods below implement the contract described in
[Model](model.md#axis-2-likelihood-how-predictors-generate-data); see
[Extending](../extending/likelihood.md) to add a new one.

## Pointwise (factorised)

| Constructor | Response \(y\) | Link | Shared parameters |
|---|---|---|---|
| [`gaussian`][lasusie.likelihoods.gaussian] | continuous | identity | \(\sigma^2\) (EB-updated) |
| [`bernoulli_logit`][lasusie.likelihoods.bernoulli_logit] | binary | logit | none |
| [`poisson_log`][lasusie.likelihoods.poisson_log] | count | log | none |
| [`neg_binomial_log`][lasusie.likelihoods.neg_binomial_log] | count (overdispersed) | log | dispersion \(r\) (fixed) |
| [`gamma_log`][lasusie.likelihoods.gamma_log] | positive continuous | log | shape \(k\) (fixed) |
| [`student_t`][lasusie.likelihoods.student_t] | continuous (heavy-tailed) | identity | d.o.f. \(\nu\), scale\(^2\) (fixed) |
| [`beta_binomial_logit`][lasusie.likelihoods.beta_binomial_logit] | proportion (overdispersed) | logit | concentration \(s\) (fixed) |
| [`ordinal_logit`][lasusie.likelihoods.ordinal_logit] | ordered categorical \(\{0,\dots,C-1\}\) | cumulative logit | \(C-1\) cutpoints (EB-updated) |
| [`ordinal_probit`][lasusie.likelihoods.ordinal_probit] | ordered categorical \(\{0,\dots,C-1\}\) | cumulative probit | \(C-1\) cutpoints (EB-updated) |
| [`aft_lognormal`][lasusie.likelihoods.aft_lognormal] | survival time, right-censored | log-normal AFT | scale (fixed) |
| [`aft_weibull`][lasusie.likelihoods.aft_weibull] | survival time, right-censored | Weibull AFT | scale (fixed) |
| [`aft_loglogistic`][lasusie.likelihoods.aft_loglogistic] | survival time, right-censored | log-logistic AFT | scale (fixed) |

These all reduce to an elementwise log-density factor decorated with
[`lasusie.model.likelihood`][lasusie.model.likelihood], which supplies
`log_density` (a sum over observations) and a Gauss-Hermite
`expected_log_density` for offset-uncertainty propagation. "Fixed" shared
parameters have no closed-form empirical-Bayes update in the current
implementation and must be supplied by the caller; the ordinal cutpoints are
"EB-updated" numerically (a Newton M-step on the expected log-likelihood
between sweeps, over a monotone reparametrisation). Because those cutpoints
absorb the outcome's baseline level, an ordinal model should not also carry a
covariate intercept.

## Composite (non-factorised)

| Constructor | Coupling | Offset-variance handling |
|---|---|---|
| [`mvn_resid`][lasusie.likelihoods.mvn_resid] | dense residual covariance across phenotypes on the same individual | exact closed form |
| [`cox`][lasusie.likelihoods.cox] | risk-set sum over other individuals (Cox proportional hazards) | diagonal delta-method (see [Algorithm](algorithm.md#approximation-2-offset-uncertainty-propagation)) |

Composite likelihoods implement `log_density`/`expected_log_density`/
`updated` directly rather than through the `likelihood` decorator, since
their log-density doesn't split into per-observation terms. See
[`MVNResidual`][lasusie.likelihoods.MVNResidual] and
[`CoxPH`][lasusie.likelihoods.CoxPH] for the underlying classes.
