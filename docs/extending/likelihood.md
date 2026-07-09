# Adding a likelihood

A likelihood only needs to answer three questions for the IBSS loop:

1. What is `log p(Y | eta)`? (`log_density`)
2. What is `E_{eta~q}[log p(Y | eta)]` under a Gaussian offset belief `q(eta)
   = N(mean, var)`? (`expected_log_density`, used when `propagate_variance=True`
   — see [Algorithm](../overview/algorithm.md#approximation-2-offset-uncertainty-propagation))
3. (optional) How should any shared parameters (e.g. residual variance) be
   updated by empirical Bayes? (`updated`)

There are two ways to implement this, matching the two shapes described in
[Model](../overview/model.md#axis-2-likelihood-how-predictors-generate-data): **pointwise**
(factorised) and **composite** (non-factorised).

## Pointwise likelihoods

Use this when `log p(Y | eta) = sum_i factor(y_i, eta_i, theta)` — the
common case for i.i.d. observations. You only need to write the elementwise
log-density `factor`; decorate it with
[`lasusie.model.likelihood`][lasusie.model.likelihood] and it supplies
`log_density` and a Gauss-Hermite `expected_log_density` for free.

```python
import jax.numpy as jnp
from lasusie.model import likelihood

@likelihood
def neg_binom(y, eta, theta):
    r = theta["r"]
    # eta is the log-mean link
    log_p = eta - jnp.logaddexp(eta, jnp.log(r))
    return (
        jax.scipy.special.gammaln(y + r) - jax.scipy.special.gammaln(r) - jax.scipy.special.gammaln(y + 1)
        + r * jnp.log(r) - r * jnp.logaddexp(eta, jnp.log(r))
        + y * log_p
    )

lik = neg_binom(y, r=2.0)
```

- `factor(y, eta, theta)` must be a pure, elementwise JAX function, since it
  is both summed for `log_density` and evaluated at quadrature nodes for
  `expected_log_density`.
- The decorator returns a [`Likelihood`][lasusie.model.Likelihood]: call it
  (`neg_binom(y, r=2.0)`) to bind data and parameters, and reach the original
  elementwise function back through `neg_binom.factor` if you need it.
- `theta` is a `dict` of shared parameters (arrays), stored as a pytree leaf
  so it's differentiable and can be updated by EB.
- To re-estimate parameters between IBSS sweeps, use the parameterized form
  `@likelihood(update_fn=...)` where
  `update_fn(y, eta_mean, eta_var, theta) -> new_theta` is the M-step (see
  `_gaussian_update` in [`lasusie.likelihoods`](../api.md) for the pattern: an
  M-step using the current latent belief's mean and variance).
- `n_quad` (default 32) controls the number of Gauss-Hermite nodes used for
  `expected_log_density`; increase it if `factor` is highly nonlinear in
  `eta`.

This is how all the built-in pointwise likelihoods (`gaussian`,
`bernoulli_logit`, `poisson_log`) are defined — see
[`lasusie/likelihoods.py`](../api.md) for the full source.

## Composite likelihoods

Use this when observations are coupled and the log-density does not split
into independent per-observation terms — e.g. a shared residual covariance
across contexts (`mvn_resid`), or a risk-set sum over other individuals
(`cox`). Subclass
[`AbstractLikelihood`][lasusie.model.AbstractLikelihood] and implement
**only `log_density`** — the base supplies composite-case defaults for
everything else (`hessian_diagonal`, `expected_log_density`, `updated`), so a
coupled likelihood works out of the box:

```python
import jax.numpy as jnp
from jax import Array

from lasusie.model import AbstractLikelihood


class MyCompositeLikelihood(AbstractLikelihood):
    # any data/parameters needed, as pytree fields
    y: Array
    theta: Array

    def log_density(self, eta: Array) -> Array:
        # full log p(Y | eta); may couple observations however you need
        ...

    # expected_log_density, hessian_diagonal, updated are inherited. Override
    # one only if your structure gives something better (see below).
```

Guidance, drawn from the two built-in composite likelihoods:

- **Override `hessian_diagonal` if your curvature has structure.** The base
  [`expected_log_density`][lasusie.model.AbstractLikelihood.expected_log_density]
  is the diagonal delta-method correction — it needs only the diagonal of the
  `log_density` Hessian, which the base
  [`hessian_diagonal`][lasusie.model.AbstractLikelihood.hessian_diagonal]
  gets by materialising the full `N x N` Hessian (`O(N^2)`). `cox`
  ([`CoxPH`][lasusie.likelihoods.CoxPH]) overrides it with an `O(N)` closed
  form (risk-set cumulative sums), and inherits the delta-method
  `expected_log_density` unchanged on top of it. The delta correction is a
  genuine approximation (it drops cross-individual offset covariance, keeping
  only each individual's *marginal* offset variance), so document that
  trade-off in your docstring the way `CoxPH` does.
- **Override `expected_log_density` directly if your coupling admits a
  closed-form offset expectation.** `mvn_resid`
  ([`MVNResidual`][lasusie.likelihoods.MVNResidual]) has a residual covariance
  across contexts on the *same* individual, so the expectation reduces to
  `log_density(mean)` minus a trace correction term — exact, and cheap, so it
  bypasses the delta method entirely.
- **Precompute anything data-derived once**, outside of `log_density`, and
  store it as a field. `CoxPH.bind` sorts individuals by descending time
  once at construction so `log_density` can use a cumulative log-sum-exp
  (`jax.lax.cumlogsumexp`) instead of recomputing risk sets — an `O(N)`
  evaluation instead of `O(N^2)`. Since the sort is `eta`-independent,
  storing it as a field means autodiff treats it as a constant, which is
  correct.
- `log_density` must be differentiable in `eta` up to second order — it is
  Laplace-approximated in effect-space by
  [`lasusie.laplace.fit`][lasusie.laplace.fit] (see
  [Algorithm](../overview/algorithm.md#approximation-1-nested-laplace-for-the-single-effect-potential)),
  which takes a gradient and a Hessian of the function passed to it, which
  ultimately calls into `log_density`/`expected_log_density` via
  `design.apply_effect`.
- Override `updated` only if you have shared parameters to refresh by
  empirical Bayes; with none, the inherited identity (`return self`) is
  already what you want (as for `CoxPH`).

## Registering a constructor

Whether pointwise or composite, expose a plain constructor function (like
`gaussian(y, sigma2=...)`, `cox(times, events)`) that binds the data and any
initial parameters, and add it to `lasusie.likelihoods` (and `__all__` in
[`lasusie/__init__.py`](../api.md) if it should be part of the public API).
The model itself doesn't need any other changes — `Model(design=...,
likelihood=my_likelihood(...), prior=..., log_pi=...)` works with any object
implementing `log_density`, `expected_log_density`, and `updated`.

## Testing a new likelihood

At minimum, verify:

- `log_density` matches a reference implementation (e.g. `scipy.stats`) on a
  handful of points.
- `expected_log_density(mean, var)` reduces to `log_density(mean)` as `var
  -> 0`.
- For pointwise likelihoods with a known closed-form expectation (e.g.
  Gaussian), `expected_log_density` matches it analytically, not just via
  quadrature agreement.
- `updated` moves shared parameters in the correct direction on synthetic
  data with a known generating parameter.

See `tests/test_likelihood.py` and `tests/test_laplace.py` in the repo for
existing patterns to follow.
