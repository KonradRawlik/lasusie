# Adding a prior

[`lasusie.model.Prior`][lasusie.model.Prior] already represents an arbitrary
mixture of zero-mean Gaussians over the effect vector \(\beta \in
\mathbb{R}^K\) (see
[Model](../overview/model.md#axis-3-prior-how-effect-sizes-are-shared-across-contexts)),
and its combine/KL/update machinery is generic over the mixture's
covariance components \(\Sigma_g\). So adding a *new sharing pattern* — the
common case — doesn't require touching `Prior` at all: it's a matter of
supplying a new set of covariance matrices.

## Adding a covariance constructor

A new prior is a plain function that builds the \(G\) covariance matrices
encoding the sharing pattern you want (e.g. "shared across half the
contexts", "hierarchical with a global and per-cluster component") and
passes them to `Prior` directly, following the pattern of the built-in
constructors:

```python
import jax.numpy as jnp
from jax import Array

from lasusie.model import Prior


def my_sharing_pattern(K: int, scale: float = 1.0) -> Array:
    """Build the (G, K, K) covariance stack for a new sharing pattern."""
    ...


def my_prior(K: int, scale: float = 1.0, weights: Array | None = None) -> Prior:
    components = my_sharing_pattern(K, scale)
    G = components.shape[0]
    log_weights = jnp.zeros(G) if weights is None else jnp.log(jnp.asarray(weights))
    return Prior(log_weights=log_weights, covariances=components)
```

- Each \(\Sigma_g\) must be positive semi-definite; a genuinely singular
  component (e.g. a "null" pattern with \(\Sigma_g \to 0\)) should be
  represented with a small ridge (see `canonical_components` in
  [`lasusie/priors.py`](../api.md)) so it stays numerically invertible where
  needed.
- Mixing weights default to uniform; pass `weights` to start from an
  informative guess (they're re-estimated by empirical Bayes between IBSS
  sweeps regardless, unless `update_prior=False`).

Add the constructor to `lasusie.priors` (and `__all__` in
[`lasusie/__init__.py`](../api.md) if it should be public).

## Beyond a Gaussian mixture

If your prior can't be expressed as a mixture of zero-mean Gaussians at all
— e.g. a prior with a nonzero mean, or a non-Gaussian component — `Prior`
itself would need a counterpart implementing the same contract the SER
relies on: combining a single-variant Gaussian potential (see
[Algorithm](../overview/algorithm.md#approximation-1-nested-laplace-for-the-single-effect-potential))
with the prior to get a posterior and a marginal likelihood, a KL to the
prior for the ELBO, and an empirical-Bayes update. This is a much larger
change than adding a covariance constructor and hasn't been needed by any of
the built-in SuSiE-family priors, so start from the covariance-constructor
route above unless your prior genuinely can't be written as a Gaussian
mixture.

## Testing a new prior

At minimum, verify:

- Each component covariance is positive semi-definite.
- `Prior.combine`/`Prior.combine_joint` on a synthetic potential produce a
  finite, positive-definite posterior for every component, including any
  singular ("null") ones.
- `Prior.update` moves the mixture weights and covariances in the correct
  direction on synthetic data with a known generating sharing pattern.

See `tests/test_prior.py` in the repo for existing patterns to follow.
