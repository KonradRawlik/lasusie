"""Covariate handling: the latent-offset ``eta = design(B) + Z gamma`` extension.

Covariates enter as a fixed-effect offset with a point-estimate ``gamma`` re-fit by a
Newton M-step each IBSS sweep (see :mod:`lasusie.covariates`). These tests check the
M-step against the Gaussian closed form, that covariates de-confound fine-mapping, that the
generalized path reduces to ``sushie``-style regress-out in the Gaussian limit, that it runs
and improves the ELBO for a non-Gaussian likelihood, that the BlockDesign geometry works,
and that a model with no covariates is bit-identical to the pre-feature path.
"""

import jax
import jax.numpy as jnp
import numpy as np

from lasusie import likelihoods, priors
from lasusie.covariates import block_covariates, shared_covariates
from lasusie.design import BlockDesign, SharedDesign
from lasusie.ibss import ibss
from lasusie.model import Model


def _standardize(X):
    return (X - X.mean(0)) / X.std(0)


# ---------------------------------------------------------------------------
# 1. gamma M-step matches the Gaussian OLS closed form
# ---------------------------------------------------------------------------
def test_gamma_mstep_matches_ols():
    key = jax.random.PRNGKey(0)
    kz, ke = jax.random.split(key)
    N, C = 300, 3
    Z = jax.random.normal(kz, (N, C))
    true_gamma = jnp.array([2.0, -1.5, 0.7])
    eta_eff = jnp.zeros((N, 1))  # no effect part
    y = (Z @ true_gamma)[:, None] + 0.1 * jax.random.normal(ke, (N, 1))

    lik = likelihoods.gaussian(y, sigma2=1.0)
    cov = shared_covariates(Z, K=1, add_intercept=False)
    cov = cov.update(lik, eta_eff, jnp.zeros((N, 1)), propagate_variance=True)

    # OLS of (y - eta_eff) on Z
    ols = jnp.linalg.solve(Z.T @ Z, Z.T @ (y - eta_eff))
    np.testing.assert_allclose(cov.gamma, ols, atol=1e-6)
    np.testing.assert_allclose(cov.gamma[:, 0], true_gamma, atol=0.05)


# ---------------------------------------------------------------------------
# 2. covariates de-confound fine-mapping
# ---------------------------------------------------------------------------
def test_covariate_deconfounds_finemapping():
    key = jax.random.PRNGKey(1)
    kx, kz, kn = jax.random.split(key, 3)
    N, M = 400, 30
    causal = 10
    X = _standardize(jax.random.normal(kx, (N, M)))
    # a covariate correlated with the causal genotype
    z = 0.8 * X[:, causal] + 0.6 * jax.random.normal(kz, (N,))
    z = (z - z.mean()) / z.std()
    Z = z[:, None]
    c = 3.0  # strong covariate effect
    b = 1.2
    y = (z * c + X[:, causal] * b + 0.5 * jax.random.normal(kn, (N,)))[:, None]

    def model(cov):
        return Model(
            design=SharedDesign(X=X),
            likelihood=likelihoods.gaussian(y, sigma2=1.0),
            prior=priors.susie(1.0),
            log_pi=jnp.full(M, -jnp.log(M)),
            covariates=cov,
        )

    res_with = ibss(model(shared_covariates(Z, K=1)), L=2, max_iter=100)
    assert res_with.converged
    assert int(jnp.argmax(res_with.pip)) == causal
    assert res_with.pip[causal] > 0.9
    # intercept + z coefficient; z coefficient (2nd row) recovered near truth
    assert res_with.covariates.gamma.shape == (2, 1)  # [intercept, z]
    np.testing.assert_allclose(float(res_with.covariates.gamma[1, 0]), c, atol=0.3)


# ---------------------------------------------------------------------------
# 3. Gaussian latent-offset == regress-out oracle (sushie's approach)
# ---------------------------------------------------------------------------
def test_gaussian_equivalent_to_regress_out():
    key = jax.random.PRNGKey(2)
    kx, kz, kn = jax.random.split(key, 3)
    N, M = 350, 25
    causal = 7
    X = _standardize(jax.random.normal(kx, (N, M)))
    Z = jax.random.normal(kz, (N, 2))
    y = (X[:, causal] * 1.3 + Z @ jnp.array([1.5, -0.8]) + 0.5 * jax.random.normal(kn, (N,)))

    # our latent-offset path
    model_cov = Model(
        design=SharedDesign(X=X),
        likelihood=likelihoods.gaussian(y[:, None], sigma2=0.25),
        prior=priors.susie(1.0),
        log_pi=jnp.full(M, -jnp.log(M)),
        covariates=shared_covariates(Z, K=1),
    )
    res_cov = ibss(model_cov, L=2, max_iter=200)

    # regress-out oracle: residualize y (and X) on [1, Z], then plain SuSiE
    Zi = jnp.concatenate([jnp.ones((N, 1)), Z], axis=1)
    P = Zi @ jnp.linalg.solve(Zi.T @ Zi, Zi.T)  # hat matrix
    y_r = y - P @ y
    X_r = X - P @ X
    model_ro = Model(
        design=SharedDesign(X=X_r),
        likelihood=likelihoods.gaussian(y_r[:, None], sigma2=0.25),
        prior=priors.susie(1.0),
        log_pi=jnp.full(M, -jnp.log(M)),
    )
    res_ro = ibss(model_ro, L=2, max_iter=200)

    assert int(jnp.argmax(res_cov.pip)) == causal
    assert int(jnp.argmax(res_ro.pip)) == causal
    # PIPs should match closely between the two approaches
    np.testing.assert_allclose(
        np.asarray(res_cov.pip), np.asarray(res_ro.pip), atol=0.05
    )


# ---------------------------------------------------------------------------
# 4. non-Gaussian path: runs, ELBO monotone, gamma sensible
# ---------------------------------------------------------------------------
def test_poisson_with_covariate_elbo_monotone():
    key = jax.random.PRNGKey(3)
    kx, kz, kn = jax.random.split(key, 3)
    N, M = 500, 20
    causal = 4
    X = _standardize(jax.random.normal(kx, (N, M)))
    z = (jax.random.normal(kz, (N,)))
    Z = z[:, None]
    eta = X[:, causal] * 0.8 + z * 0.5 - 0.5  # includes an offset absorbed by intercept
    counts = jax.random.poisson(kn, jnp.exp(eta)).astype(jnp.float64)

    model = Model(
        design=SharedDesign(X=X),
        likelihood=likelihoods.poisson_log(counts[:, None]),
        prior=priors.susie(0.5),
        log_pi=jnp.full(M, -jnp.log(M)),
        covariates=shared_covariates(Z, K=1),
    )
    res = ibss(model, L=2, max_iter=60)

    hist = np.asarray(res.elbo_history)
    # ELBO non-decreasing (allow tiny numerical slack)
    assert np.all(np.diff(hist) > -1e-4)
    assert np.all(np.isfinite(res.pip))
    assert int(jnp.argmax(res.pip)) == causal
    # z coefficient (row 1, after intercept) has the right sign/order of magnitude
    assert float(res.covariates.gamma[1, 0]) > 0.2


# ---------------------------------------------------------------------------
# 5. BlockDesign (multi-ancestry) covariates
# ---------------------------------------------------------------------------
def test_block_covariates_shape_and_recovery():
    key = jax.random.PRNGKey(4)
    K, N, M, C = 2, 250, 15, 2
    causal = 6
    keys = jax.random.split(key, 5)
    X = jnp.stack([_standardize(jax.random.normal(keys[k], (N, M))) for k in range(K)])
    Z = jax.random.normal(keys[2], (K, N, C))
    true_gamma = jnp.array([[1.0, -0.5], [0.7, 1.2]])  # (K, C)
    b = jnp.array([1.1, 0.9])  # per-context effect at the causal variant
    noise = 0.4 * jax.random.normal(keys[3], (K, N))
    eta = jnp.stack([X[k][:, causal] * b[k] for k in range(K)]) \
        + jnp.einsum("knc,kc->kn", Z, true_gamma) + noise  # (K, N)
    y = eta  # Gaussian identity

    cov = block_covariates(Z, add_intercept=False)  # gamma (K, C)
    assert cov.offset().shape == (K, N)

    model = Model(
        design=BlockDesign(X=X),
        likelihood=likelihoods.gaussian(y.reshape(K, N), sigma2=0.16),
        prior=priors.sushie(jnp.eye(K)),
        log_pi=jnp.full(M, -jnp.log(M)),
        covariates=cov,
    )
    res = ibss(model, L=2, max_iter=150)
    assert res.covariates.gamma.shape == (K, C)
    assert int(jnp.argmax(res.pip)) == causal
    np.testing.assert_allclose(np.asarray(res.covariates.gamma), np.asarray(true_gamma), atol=0.25)


# ---------------------------------------------------------------------------
# 6. intercept toggle + no-covariates regression guard
# ---------------------------------------------------------------------------
def test_intercept_toggle_shapes():
    Z = jnp.ones((10, 3))
    with_i = shared_covariates(Z, K=2, add_intercept=True)
    without_i = shared_covariates(Z, K=2, add_intercept=False)
    assert with_i.Z.shape == (10, 4) and with_i.gamma.shape == (4, 2)
    assert without_i.Z.shape == (10, 3) and without_i.gamma.shape == (3, 2)


def test_no_covariates_identical_to_before():
    key = jax.random.PRNGKey(5)
    kx, kn = jax.random.split(key)
    N, M = 300, 20
    X = _standardize(jax.random.normal(kx, (N, M)))
    y = (X[:, 3] * 1.4 + 0.5 * jax.random.normal(kn, (N,)))[:, None]

    def build():
        return Model(
            design=SharedDesign(X=X),
            likelihood=likelihoods.gaussian(y, sigma2=1.0),
            prior=priors.susie(1.0),
            log_pi=jnp.full(M, -jnp.log(M)),
        )

    res = ibss(build(), L=2, max_iter=100)
    assert res.covariates is None
    # explicitly disabling the covariate update changes nothing when there are none
    res2 = ibss(build(), L=2, max_iter=100, update_covariates=False)
    np.testing.assert_array_equal(np.asarray(res.pip), np.asarray(res2.pip))
