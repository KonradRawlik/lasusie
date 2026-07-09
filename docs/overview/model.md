# Model

`lasusie` fits a single family of models, parameterized along three
orthogonal axes: the **design**, the **likelihood**, and the **prior**. This
page describes the model class those axes span; see [Algorithm](algorithm.md)
for how inference is done, and [Extending](../extending/index.md) for how to add
new components.

## The general latent linear model with a SuSiE prior

We observe data \(\mathcal{Y}\) that follows, conditional on \(K\) latent
linear predictors \(\eta_{1\dots K} \in \mathbb{R}^N\), some distribution
\(\phi\):

\[
\begin{aligned}
\mathcal{Y} &\sim \phi(\eta_{1\dots K}, \theta_\phi) \\
\eta_k &= \mathbf{X}_k \sum_l \mathbf{b}_{k,l} \\
\mathbf{b}_{k,l} &= \mathbf{\alpha}_l \cdot \beta_{k,l} \\
\mathbf{\alpha}_l &\sim \mathrm{Mult}(1, \bar\alpha), \qquad
\mathbf{\beta}_l = [\beta_{1,l}, \dots, \beta_{K,l}] \sim \pi(\theta_\pi)
\end{aligned}
\]

Each latent predictor \(\eta_k\) is built from a predictor-specific design
matrix \(\mathbf{X}_k \in \mathbb{R}^{N\times M}\) and a sum of \(L\) **single
effects**. Each single effect \(l\) picks *one* active variant out of \(M\)
candidates via a one-hot indicator \(\mathbf{\alpha}_l\) (shared across
all \(K\) contexts), and gives it a context-specific effect size
\(\beta_{k,l}\) drawn from a joint prior \(\pi\) over the \(K\)-vector
\(\mathbf{\beta}_l\). This is the defining structure of SuSiE: sparsity
(one active variant per effect) is shared across contexts, but the
*magnitude* of the effect is allowed to vary or covary across them.

The original univariate SuSiE model is the special case \(\mathcal{Y} =
\mathbf{y}\), \(K=1\), \(\phi = \mathcal{N}(y; \eta_1, \sigma^2)\), and
\(\pi(\cdot) = \mathcal{N}(\cdot; 0, \sigma_0^2)\).

!!! note "In the code"
    This whole specification is a [`lasusie.model.Model`][lasusie.model.Model]:
    a `design`, a `likelihood`, and a `prior`, plus per-variant inclusion
    weights `log_pi` and an optional `covariates` offset (below).

### Covariates

Nuisance covariates \(Z\) (intercept, age, sex, genotype PCs, batch) enter as
an additive **fixed-effect offset** to the latent predictor:

\[
\eta_k = X_k \sum_l \mathbf{b}_{k,l} + Z_k\,\gamma_k .
\]

The coefficients \(\gamma\) are a **point estimate** — a maximum-(expected-)
likelihood fixed effect with no prior and no propagated variance, the standard
treatment for nuisance covariates. Because it is a plain maximization rather
than a Bayesian update, it composes with the coordinate-ascent loop as an
additional M-step alongside the prior and likelihood updates: re-estimate
\(\gamma\) with the effect belief held fixed, then continue. This
latent-offset form is likelihood-agnostic — it works for every likelihood,
whereas OLS regress-out is only exact under a Gaussian likelihood — and
reduces exactly to regress-out in the Gaussian limit. The offset's geometry
mirrors the design axis below: one shared \(Z\) with a per-context
\(\gamma_k\), or a per-context \(Z_k\) each with its own \(\gamma_k\),
matching whichever of the two design configurations is in use.

!!! note "In the code"
    [`shared_covariates`][lasusie.covariates.shared_covariates] builds the
    shared-\(Z\) form (for `SharedDesign`);
    [`block_covariates`][lasusie.covariates.block_covariates] builds the
    per-context form (for `BlockDesign`). Both add an intercept column by
    default. Pass the result as `Model(..., covariates=...)`; the fitted
    \(\gamma\) is returned as `FineMapResult.covariate_coef`.

## Axis 1: design — how effects become predictors

The design axis specifies the linear map \(\eta = X(B)\) from the \(M \times
K\) effect array \(B\) to the latent predictor(s). Multi-ancestry and
multi-phenotype fine-mapping are two configurations of the *same* operator,
differing only in whether the \(N \times M\) genotype matrix is shared across
contexts or specific to each:

- **Shared genotypes** (multi-phenotype): one design matrix \(X \in
  \mathbb{R}^{N \times M}\), shared across all \(K\) contexts, every
  individual measured in every context. \(\eta = XB\) has shape \((N, K)\).
- **Block genotypes** (multi-ancestry): disjoint individuals per context,
  with a stacked design \(X\) of shape \((K, N, M)\). \(\eta_k = X_k
  B_{:,k}\), shape \((K, N)\).

\(K=1\) univariate SuSiE is just the shared-genotype case with a single
context.

!!! note "In the code"
    A [`lasusie.design.DesignMap`][lasusie.design.DesignMap] implements this
    map: [`SharedDesign`][lasusie.design.SharedDesign] for the shared case,
    [`BlockDesign`][lasusie.design.BlockDesign] for the block case.

## Axis 2: likelihood — how predictors generate data

The likelihood axis specifies \(\phi\), the observation model relating
\(\eta\) to the data \(\mathcal{Y}\). Two shapes of likelihood are supported,
distinguished by whether the log-density decomposes observation-by-observation:

- **Pointwise (factorised)**: \(\log p(Y \mid \eta) = \sum_i
  \mathrm{factor}(y_i, \eta_i, \theta)\), a sum of independent per-observation
  terms.
- **Composite (non-factorised)**: observations are coupled, so the
  log-density cannot be written as such a sum — e.g. a shared covariance
  linking several observations, or a term that depends on a summary of the
  whole dataset rather than just one observation.

The distinction matters for inference: pointwise likelihoods admit an exact,
cheap expectation over offset uncertainty; composite likelihoods need a more
general (but approximate) correction. See
[Algorithm](algorithm.md#approximation-2-offset-uncertainty-propagation) for
the mechanics.

!!! note "In the code"
    Defined in [`lasusie.likelihoods`](../api.md) — see
    [Built-in likelihoods](likelihoods.md) for the full list, or
    [Extending](../extending/likelihood.md) to add a new one.

## Axis 3: prior — how effect sizes are shared across contexts

The prior axis specifies \(\pi(\theta_\pi)\), the joint distribution over the
\(K\)-vector \(\mathbf{\beta}_l\) for a single effect, represented uniformly
as a mixture of zero-mean Gaussians,

\[
\pi(\beta) = \sum_g w_g\, \mathcal{N}(\beta; 0, \Sigma_g),
\]

over a fixed set of covariance "sharing patterns" \(\Sigma_g\). This single
mixture representation subsumes the whole SuSiE family:

| Method | \(K\) | Mixture |
|---|---|---|
| univariate SuSiE | 1 | one component \(\mathcal{N}(0, \sigma_0^2)\) |
| sushie (multi-ancestry) | >1 | one dense component \(\mathcal{N}(0, \Sigma)\) |
| mvSuSiE (multi-phenotype) | >1 | mixture of sharing patterns (shared, independent, context-specific, null) |
| point-normal | any | a component with \(\Sigma \to 0\) |

Because the prior is conjugate to the Gaussian potential produced by the
likelihood approximation step (see [Algorithm](algorithm.md)), combining a
prior component with a potential is closed-form regardless of which of the
above you pick — the mixture is what lets a single code path serve all of
them.

!!! note "In the code"
    A [`lasusie.model.Prior`][lasusie.model.Prior], built by the constructors
    in [`lasusie.priors`](../api.md): [`susie`][lasusie.priors.susie],
    [`sushie`][lasusie.priors.sushie], [`mvsusie`][lasusie.priors.mvsusie];
    the point-normal component is included via `canonical_components`.

!!! note "Putting it together"
    ```python
    import lasusie as ls

    model = ls.Model(
        design=ls.SharedDesign(X=X),                 # M x K design
        likelihood=ls.likelihoods.gaussian(y, sigma2=1.0),
        prior=ls.priors.susie(sigma0_sq=0.01),
        log_pi=jnp.full((M,), -jnp.log(M)),           # uniform inclusion prior
    )

    result = ls.finemap(model, L=10)
    ```
    `finemap` runs inference (the IBSS loop, see [Algorithm](algorithm.md)) and
    returns PIPs and credible sets.
