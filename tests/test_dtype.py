"""Compute-dtype handling: inference, the explicit ``dtype=`` override, and the guard.

The library resolves one floating precision per run and casts the whole model to it, so no
array silently promotes. The suite runs in float64 (conftest); these tests force float32 to
check the *other* precision genuinely works end-to-end -- the joint fit, the EB M-step, the
covariate Newton step and the leaf ridges all build ``eye``/``ladder`` arrays that used to
default to the global config dtype and would mismatch a float32 carry.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lasusie import Model, finemap, likelihoods, priors, shared_covariates
from lasusie.design import SharedDesign
from lasusie.priors import canonical_components


def _gaussian_mixture_model():
    rng = np.random.default_rng(0)
    N, M, j, K = 300, 40, 12, 2
    X = rng.standard_normal((N, M))
    X = (X - X.mean(0)) / X.std(0)
    Y = np.outer(X[:, j], [1.3, 1.0]) + rng.standard_normal((N, K))
    model = Model(
        design=SharedDesign(X=jnp.asarray(X)),
        likelihood=likelihoods.mvn_resid(jnp.asarray(Y)),
        prior=priors.mvsusie(canonical_components(K=K)),
        log_pi=jnp.full(M, -jnp.log(M)),
    )
    return model, j


def test_explicit_float32_runs_in_float32_and_recovers():
    """Forcing float32 under the suite's x64 config must actually compute in float32 -- cast
    the model + every internal array -- without promoting or erroring, and still fine-map."""
    model, j = _gaussian_mixture_model()
    res = finemap(model, L=3, update_prior=True, ser_fit="joint", dtype=jnp.float32)
    assert res.pip.dtype == np.float32
    assert res.posterior.mean.dtype == jnp.float32  # internal state cast too
    assert not np.isnan(res.pip).any()
    assert res.pip[j] > 0.9


def test_default_dtype_inferred_from_model():
    """``dtype=None`` takes the widest of the model's arrays; under x64 that is float64."""
    model, _ = _gaussian_mixture_model()
    res = finemap(model, L=3, dtype=None)
    assert res.pip.dtype == np.float64


def test_float32_survives_covariate_mstep_and_count_likelihood():
    """The float32 path must clear the covariate Newton M-step and a count likelihood -- the
    leaf helpers (``damped_newton`` ladder/eye, ``laplace.fit`` ridge) that build derived
    arrays -- without a dtype-mismatch ``scan`` carry error."""
    rng = np.random.default_rng(1)
    N, M, j, K = 250, 30, 8, 2
    X = rng.standard_normal((N, M))
    X = X - X.mean(0)
    mu = np.exp(1.2 + np.outer(X[:, j], [1.0, 0.8]))
    Y = rng.poisson(mu).astype(float)
    model = Model(
        design=SharedDesign(X=jnp.asarray(X)),
        likelihood=likelihoods.neg_binomial_log(jnp.asarray(Y), r=3.0),
        prior=priors.mvsusie(canonical_components(K=K)),
        log_pi=jnp.full(M, -jnp.log(M)),
        covariates=shared_covariates(jnp.ones((N, 1)), K=K, add_intercept=False),
    )
    res = finemap(model, L=5, update_prior=True, ser_fit="joint", dtype=jnp.float32)
    assert res.pip.dtype == np.float32
    assert res.covariate_coef.dtype == np.float32
    assert not np.isnan(res.pip).any()


def test_requesting_float64_without_x64_raises():
    """An explicit 64-bit request that JAX would silently canonicalise to 32-bit must raise,
    not quietly downcast -- the whole point of the resolver is to remove that silent trap."""
    model, _ = _gaussian_mixture_model()
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="jax_enable_x64"):
            finemap(model, L=2, dtype=jnp.float64)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_model_carrying_float64_without_x64_raises():
    """The guard also fires on inference (``dtype=None``): a model built under x64 that still
    holds float64 arrays after the flag is turned off would be silently downcast -- catch it."""
    model, _ = _gaussian_mixture_model()  # built under the suite's x64 -> float64 leaves
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="the model holds float64 arrays"):
            finemap(model, L=2, dtype=None)
    finally:
        jax.config.update("jax_enable_x64", True)
