# Algorithm

This page explains how `lasusie` performs inference in the generalized SuSiE
model described in [Model](model.md), and the two independent
approximations the implementation makes to get there. Each section describes
the algorithm in the abstract first; a note at the end of the section points
to where it lives in the code.

## Why approximate inference is needed

In the original Gaussian, univariate SuSiE model, the posterior over a
single effect is available in closed form, which is what makes the SuSiE
coordinate-ascent algorithm exact in that special case. Once the likelihood
is non-Gaussian, or the effect is a vector coupled across contexts by a
non-diagonal prior, closed-form inference is not possible even for a
*single* effect. `lasusie` uses variational inference with two independent
approximations to recover a SuSiE-like algorithm in this general setting.

## Step-wise variational inference (IBSS)

We posit a mean-field variational family that factorises across the \(L\)
effects:

\[
q(\alpha, \beta) = \prod_l q_l(\alpha_l, \beta_l), \qquad
q_l(\alpha_l, \beta_l) = \mathrm{Mult}(\alpha_l; 1, \bar\alpha_l)
\prod_m \mathcal{N}(\beta_l; \mu_{l,m}, \Sigma_{l,m})^{\alpha_{l,m}}
\]

i.e. each effect has a categorical belief over *which* variant is active,
and a per-candidate-variant Gaussian belief over the effect size. The ELBO is

\[
\mathcal{L} = \mathbb{E}_q[\log p(y, \alpha, \beta)] + H[q].
\]

Optimizing \(q_l\) with the other effects held fixed is a **single-effect
regression (SER)**: a Bayesian regression of the *residualized* data (data
with every other effect's current contribution subtracted out) against one
active variant. Iterative Bayesian Stepwise Selection (IBSS) is coordinate
ascent on the ELBO over the \(L\) effects: sweep over \(l = 1, \dots, L\),
each time removing effect \(l\)'s current contribution from the latent
belief \(q(\eta) = \mathcal{N}(\text{mean}, \text{var})\), re-fitting its SER
against that offset, and adding the updated effect back.

Between sweeps, empirical-Bayes M-steps optionally re-estimate each effect's
prior, any shared likelihood parameters (e.g. residual variance), and any
covariate coefficients. The covariate step is a Newton maximization of the
same expected-log-likelihood term the ELBO monitors, with the effect belief
held fixed and the coefficients treated as a point estimate — a
coordinate-ascent step that never decreases the ELBO. Convergence is
monitored on the ELBO.

!!! note "In the code"
    The outer loop is [`ibss`][lasusie.ibss.ibss]; the per-effect SER step is
    [`single_effect_regression`][lasusie.ibss.single_effect_regression]. The
    three M-steps are toggled independently by `update_prior`,
    `update_likelihood`, and `update_covariates` on `ibss`.

## Approximation 1: nested Laplace for the single-effect potential

Within one SER, for each candidate variant we need a posterior over the
effect \(\beta \in \mathbb{R}^K\) that combines the likelihood's evidence
about \(\beta\) with the prior \(\pi(\beta) = \sum_g w_g \mathcal{N}(\beta;
0, \Sigma_g)\). In general the likelihood's evidence,
\(g(\beta) = \mathbb{E}_q[\log p(Y \mid \eta(\beta))]\), is not Gaussian in
\(\beta\), so a Laplace approximation is needed somewhere in this step;
*where exactly* it gets applied is a genuine design choice with a real
trade-off.

### Mode-finding

Both choices below reduce to the same primitive: Laplace-approximate some
scalar objective \(h: \mathbb{R}^K \to \mathbb{R}\) around its mode, i.e.
locate \(\hat\beta = \arg\max_\beta h(\beta)\) and match curvature there,
giving the Gaussian approximation

\[
h(\beta) \approx h(\hat\beta) - \tfrac{1}{2}(\beta - \hat\beta)^T P (\beta - \hat\beta),
\qquad P = -\nabla^2 h(\hat\beta).
\]

Because \(\beta\) is only \(K\)-dimensional (the number of contexts —
small), this mode-finding and curvature computation is cheap regardless of
the sample size or the complexity of the likelihood. It is applied once per
candidate variant per sweep (and, for the joint fit below, once per prior
component too — "nested" inside the outer coordinate-ascent loop).

!!! note "In the code"
    The Gaussian approximation is a [`GaussianPotential`][lasusie.laplace.GaussianPotential]
    (`log_scale` \(= h(\hat\beta)\), `mode` \(=\hat\beta\), `precision` \(=P\)),
    produced by [`laplace.fit`][lasusie.laplace.fit]. For speed under
    `jit`/`vmap` across all variants, it uses a frozen-metric (chord)
    iteration rather than full Newton: the inverse Hessian is computed once
    at a warm-started initial point and reused as a fixed preconditioner for
    a fixed number of gradient-ascent steps, then the curvature is
    recomputed once more at the converged mode. This trades one Hessian
    evaluation per Newton step for one total (and avoids a data-dependent
    `while` loop), at the cost of assuming the Hessian doesn't vary too much
    between the warm start and the mode — the regime in which a Laplace
    approximation is accurate in the first place.

### Combining the potential with the prior

The remaining question is *what* gets Laplace-approximated: the likelihood
alone, or the likelihood together with each prior component.

- **Joint fit (the default).** Laplace-approximate the **joint**
  \(h_g(\beta) = g(\beta) + \log \mathcal{N}(\beta; 0, \Sigma_g)\) separately
  for each prior component \(g\). The prior curvature \(\Sigma_g^{-1}\) is
  folded in *before* finding the mode, so it regularises the objective at
  the source: for count and other exponential-family likelihoods (e.g.
  Poisson, negative binomial, gamma regression), whose log-density grows
  only *linearly* in the tail, a null (quasi-separated) variant makes
  \(g(\beta)\) monotone with no interior maximum — but \(\Sigma_g^{-1}\)
  makes every \(h_g\) strictly concave regardless, so the mode stays finite
  and the precision positive-definite. This costs one Laplace fit *per
  mixture component* rather than one total (no extra cost for
  single-component priors), and it reduces exactly to the conjugate fit
  below for a Gaussian likelihood.
- **Conjugate fit (cheaper, opt-in).** Laplace-approximate the likelihood
  \(g(\beta)\) *alone*, once, and combine the resulting Gaussian potential
  with each prior component in closed form — exact, because a mixture of
  zero-mean Gaussians is conjugate to a Gaussian potential. This is what
  lets a single code path serve the whole SuSiE family (univariate, sushie,
  mvSuSiE): only the *shape* of the prior mixture differs between them. It
  is cheaper (one fit instead of one per component), but inherits the joint
  fit's failure mode above without the prior curvature to protect it, so it
  needs numerical safeguards (mode clipping, a precision floor) to degrade
  gracefully instead of diverging.

!!! note "In the code"
    Selected via `ser_fit` on
    [`single_effect_regression`][lasusie.ibss.single_effect_regression]:
    `"joint"` (default) uses
    [`Prior.combine_joint`][lasusie.model.Prior.combine_joint]; `"conjugate"`
    uses [`Prior.combine`][lasusie.model.Prior.combine]. The numerical
    safeguards for the conjugate path live in
    [`laplace.fit`][lasusie.laplace.fit] and are inert for a
    well-identified Gaussian mode.

## Approximation 2: offset-uncertainty propagation

The second approximation concerns how much of the *other* effects'
uncertainty a single-effect regression sees. When residualizing for effect
\(l\), the offset belief \(q(\eta_{\neg l})\) has both a mean and a
variance (from the other effects' posteriors); the SER should evaluate an
*expectation* of the log-likelihood under that variance, not just plug in
the mean. Two levels of approximation are available:

- **Zeroth order.** Use the log-likelihood at the offset mean only, ignoring
  its variance entirely. Cheaper, but ignores uncertainty contributed by the
  other effects — a coarser approximation.
- **Variance-propagating (the default).** Use the expected log-likelihood
  under the offset's Gaussian belief. This is what recovers SuSiE's *exact*
  behaviour on the original Gaussian, univariate special case, since there
  the expectation is handled exactly.

How that expectation is computed depends on the likelihood's shape (see
[Model](model.md#axis-2-likelihood-how-predictors-generate-data)):

- **Pointwise (factorised) likelihoods** compute the expectation *exactly*
  for a Gaussian likelihood, and to high accuracy otherwise, via elementwise
  **Gauss-Hermite quadrature**: each observation's expectation is a 1-D
  integral, evaluated as a weighted sum over quadrature nodes.
- **Composite (non-factorised) likelihoods** don't split into
  per-observation terms, so no per-observation quadrature is possible. They
  default to a second-order (delta-method) correction under a
  diagonal-covariance offset belief:

\[
\mathbb{E}_q[\log p] \approx \log p(\text{mean}) + \tfrac{1}{2}
\sum_n H_{nn}(\text{mean})\, \text{var}_n,
\]

where \(H\) is the Hessian of the log-density at the mean. This accounts
for each observation's *marginal* offset variance but not cross-observation
covariance — a deliberate compromise, since computing the correction's
value only (not differentiating through it) avoids needing third- and
fourth-order derivatives for a term that is already a higher-order
correction than the rest of the approximation tracks.

The two approximations compose: the Laplace step (Approximation 1) handles
the (possibly non-Gaussian) map from \(\beta\) to a log-density, while
variance propagation handles how much of the *other* effects' uncertainty
that log-density is evaluated under. Turning variance propagation off also
switches the ELBO's expected-log-likelihood term to the zeroth-order form,
so the quantity being monitored always matches what was actually optimized.

!!! note "In the code"
    Controlled by `propagate_variance` on
    [`ibss`][lasusie.ibss.ibss]/[`single_effect_regression`][lasusie.ibss.single_effect_regression],
    which switches between `likelihood.log_density` (zeroth order) and
    `likelihood.expected_log_density` (variance-propagating). Gauss-Hermite
    quadrature is [`PointwiseLikelihood.expected_log_density`][lasusie.model.PointwiseLikelihood.expected_log_density];
    the delta-method default is
    [`AbstractLikelihood.expected_log_density`][lasusie.model.AbstractLikelihood.expected_log_density],
    used by composite likelihoods such as `cox` (which supplies a structured
    `O(N)` `hessian_diagonal` for the correction). `mvn_resid` uses the same
    diagonal-offset idea but has its own exact closed form, since its
    coupling (across phenotypes, not across individuals) is already
    accounted for in the residual covariance.

## Output

Fine-mapping turns the per-effect inclusion probabilities into posterior
inclusion probabilities (PIPs) and credible sets, using a purity filter
(minimum pairwise genotype correlation within a set) to decide which
credible sets to keep.

!!! note "In the code"
    [`finemap`][lasusie.api.finemap] wraps `ibss` and builds
    [`CredibleSet`][lasusie.api.CredibleSet]s with the purity filter, a lean
    port of `sushie`'s `make_cs`.
