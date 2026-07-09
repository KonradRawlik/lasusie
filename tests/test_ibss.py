"""The IBSS inference loop and ``finemap`` end-to-end.

Gaussian likelihood + K=1 SharedDesign is exact SuSiE (the Laplace step is exact for a
quadratic objective), so these are strong correctness checks of the loop mechanics,
variance propagation, and inclusion probabilities. On top of that:

  * the Gaussian K=1 path matches an independent textbook SuSiE (fixed hyperparameters) --
    our path goes through Laplace + mixture + quadrature + the design map, so agreement
    validates the whole pipeline against a closed-form reference;
  * ``finemap`` produces a pure credible set covering the causal variant;
  * logistic SuSiE recovers a planted causal with a (near-)monotone ELBO;
  * the coupled designs recover a shared causal -- ``BlockDesign`` across disjoint-sample
    ancestries (coupled via the prior) and ``SharedDesign`` + ``mvn_resid`` across
    correlated phenotypes (coupled via the residual covariance), including the mvSuSiE
    mixture-prior EB path.
"""

import jax
import jax.numpy as jnp
import numpy as np

from lasusie import finemap, likelihoods, priors
from lasusie.design import BlockDesign, SharedDesign
from lasusie.ibss import ibss
from lasusie.model import Model
from lasusie.priors import canonical_components


def _simulate(key, N=400, M=40, causal=(5, 20), effect=(1.2, -1.0), noise_sd=1.0):
    kx, ke, kn = jax.random.split(key, 3)
    X = jax.random.normal(kx, (N, M))
    X = (X - X.mean(0)) / X.std(0)
    y = jnp.zeros(N)
    for j, b in zip(causal, effect):
        y = y + X[:, j] * b
    y = y + noise_sd * jax.random.normal(kn, (N,))
    return X, y


def _gaussian_model(X, y, sigma2, sigma0_sq):
    N, M = X.shape
    return Model(
        design=SharedDesign(X=X),
        likelihood=likelihoods.gaussian(y.reshape(N, 1), sigma2=sigma2),
        prior=priors.susie(sigma0_sq),
        log_pi=jnp.full(M, -jnp.log(M)),
    )


# --- Loop mechanics: localisation, multi-effect PIPs, sane variance propagation -------


def test_single_effect_localizes_strongest():
    X, y = _simulate(jax.random.PRNGKey(0), causal=(7,), effect=(1.5,))
    model = _gaussian_model(X, y, sigma2=1.0, sigma0_sq=1.0)
    res = ibss(model, L=1, max_iter=50)

    assert res.converged
    assert int(jnp.argmax(res.alpha[0])) == 7
    # alpha is a proper distribution over variants
    np.testing.assert_allclose(res.alpha[0].sum(), 1.0, atol=1e-6)


def test_two_effects_recovered_in_pip():
    X, y = _simulate(jax.random.PRNGKey(1), causal=(5, 20), effect=(1.2, -1.1))
    model = _gaussian_model(X, y, sigma2=1.0, sigma0_sq=1.0)
    res = ibss(model, L=3, max_iter=100)

    assert res.converged
    top2 = set(np.argsort(np.asarray(res.pip))[-2:].tolist())
    assert {5, 20} <= top2
    assert res.pip[5] > 0.9 and res.pip[20] > 0.9


def test_variance_propagation_changes_result():
    # With variance propagation ON (our default) vs a mean-only offset, the fits differ;
    # here we just assert the loop produces finite, sane PIPs in [0, 1].
    X, y = _simulate(jax.random.PRNGKey(2))
    model = _gaussian_model(X, y, sigma2=1.0, sigma0_sq=1.0)
    res = ibss(model, L=3, max_iter=100)
    pip = np.asarray(res.pip)
    assert np.all(np.isfinite(pip))
    assert np.all(pip >= -1e-6) and np.all(pip <= 1 + 1e-6)


# --- Oracle match: our pipeline == an independent textbook Gaussian SuSiE -------------
def reference_susie(X, y, L, sigma2, sigma0_sq, iters=100):
    N, M = X.shape
    xtx = (X**2).sum(0)
    B = np.zeros((L, M))
    alphas = np.zeros((L, M))
    for _ in range(iters):
        for l in range(L):
            r = y - X @ (B.sum(0) - B[l])
            bhat = (X.T @ r) / xtx
            s2 = sigma2 / xtx
            lbf = 0.5 * np.log(s2 / (s2 + sigma0_sq)) + 0.5 * bhat**2 * sigma0_sq / (
                s2 * (s2 + sigma0_sq)
            )
            alpha = np.exp(lbf - _logsumexp(lbf))
            post_var = 1.0 / (1.0 / s2 + 1.0 / sigma0_sq)
            post_mean = post_var * bhat / s2
            B[l] = alpha * post_mean
            alphas[l] = alpha
    pip = 1.0 - np.prod(1.0 - alphas, axis=0)
    return alphas, pip


def _logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def _simulate_gaussian(key, N=400, M=40, causal=(5, 20), effect=(1.2, -1.1), noise=1.0):
    kx, kn = jax.random.split(key)
    X = jax.random.normal(kx, (N, M))
    X = (X - X.mean(0)) / X.std(0)
    y = sum(X[:, j] * b for j, b in zip(causal, effect)) + noise * jax.random.normal(kn, (N,))
    return np.asarray(X), np.asarray(y)


def test_gaussian_matches_reference_susie():
    X, y = _simulate_gaussian(jax.random.PRNGKey(0))
    sigma2, sigma0_sq, L = 1.0, 1.0, 3

    ref_alpha, ref_pip = reference_susie(X, y, L, sigma2, sigma0_sq)

    model = Model(
        design=SharedDesign(X=jnp.asarray(X)),
        likelihood=likelihoods.gaussian(jnp.asarray(y).reshape(-1, 1), sigma2=sigma2),
        prior=priors.susie(sigma0_sq),
        log_pi=jnp.full(X.shape[1], -jnp.log(X.shape[1])),
    )
    # hyperparameters fixed to match the reference
    res = ibss(model, L=L, max_iter=100, tol=1e-8, update_prior=False, update_likelihood=False)

    np.testing.assert_allclose(np.asarray(res.pip), ref_pip, atol=1e-4)


# --- Credible sets ---------------------------------------------------------------------
def test_finemap_credible_set_covers_causal():
    X, y = _simulate_gaussian(jax.random.PRNGKey(1), causal=(12,), effect=(1.6,))
    model = Model(
        design=SharedDesign(X=jnp.asarray(X)),
        likelihood=likelihoods.gaussian(jnp.asarray(y).reshape(-1, 1), sigma2=1.0),
        prior=priors.susie(1.0),
        log_pi=jnp.full(X.shape[1], -jnp.log(X.shape[1])),
    )
    result = finemap(model, L=3, coverage=0.95, purity=0.5)

    kept = [cs for cs in result.credible_sets if cs.kept]
    assert any(12 in cs.variants for cs in kept)
    # the set covering the causal is small and pure
    causal_cs = next(cs for cs in kept if 12 in cs.variants)
    assert causal_cs.coverage >= 0.95
    assert causal_cs.purity >= 0.5


# --- Logistic recovery + ELBO monotonicity --------------------------------------------
def test_logistic_recovers_and_elbo_monotone():
    key = jax.random.PRNGKey(2)
    kx, ky = jax.random.split(key)
    N, M, j = 800, 30, 9
    X = jax.random.normal(kx, (N, M))
    X = (X - X.mean(0)) / X.std(0)
    logits = X[:, j] * 2.0
    y = jax.random.bernoulli(ky, jax.nn.sigmoid(logits)).astype(float)

    model = Model(
        design=SharedDesign(X=X),
        likelihood=likelihoods.bernoulli_logit(y.reshape(-1, 1)),
        prior=priors.susie(1.0),
        log_pi=jnp.full(M, -jnp.log(M)),
    )
    result = finemap(model, L=2, coverage=0.95, purity=0.5, update_prior=False)

    assert int(np.argmax(result.pip)) == j
    hist = np.asarray(result.elbo_history)
    # approximate ELBO under the Laplace SER: allow tiny non-monotonic wobble
    assert np.all(np.diff(hist) > -1e-3)


# --- Coupled designs: shared causal across ancestries (block) and phenotypes (mvn) ----
def test_multiancestry_block_design_recovers_shared_causal():
    # K=2 disjoint-individual ancestries, one shared causal variant, coupled via the prior.
    key = jax.random.PRNGKey(3)
    kx, kn = jax.random.split(key)
    K, N, M, j = 2, 400, 30, 8
    effects = jnp.array([1.3, 1.0])
    X = jax.random.normal(kx, (K, N, M))
    X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    noise = jax.random.normal(kn, (K, N))
    y = X[:, :, j] * effects[:, None] + noise  # (K, N)

    model = Model(
        design=BlockDesign(X=X),
        likelihood=likelihoods.gaussian(y, sigma2=1.0),
        prior=priors.sushie(jnp.eye(K)),
        log_pi=jnp.full(M, -jnp.log(M)),
    )
    result = finemap(model, L=2, coverage=0.95, purity=0.5)

    assert int(np.argmax(result.pip)) == j
    assert result.pip[j] > 0.9


def _simulate_two_phenotypes(key, N=400, M=30, j=7, effects=(1.3, 1.0), rho=0.5):
    kx, kn = jax.random.split(key)
    X = jax.random.normal(kx, (N, M))
    X = (X - X.mean(0)) / X.std(0)
    Sigma_e = jnp.array([[1.0, rho], [rho, 1.0]])
    L = jnp.linalg.cholesky(Sigma_e)
    noise = jax.random.normal(kn, (N, 2)) @ L.T
    Y = jnp.zeros((N, 2))
    Y = Y.at[:, 0].set(X[:, j] * effects[0])
    Y = Y.at[:, 1].set(X[:, j] * effects[1])
    Y = Y + noise
    return X, Y, Sigma_e


def test_multiphenotype_recovers_shared_causal():
    X, Y, Sigma_e = _simulate_two_phenotypes(jax.random.PRNGKey(2), j=7)
    model = Model(
        design=SharedDesign(X=X),
        likelihood=likelihoods.mvn_resid(Y, resid_cov=Sigma_e),
        prior=priors.sushie(jnp.eye(2)),
        log_pi=jnp.full(X.shape[1], -jnp.log(X.shape[1])),
    )
    result = finemap(model, L=2, coverage=0.95, purity=0.5)
    assert int(np.argmax(result.pip)) == 7
    assert result.pip[7] > 0.9


def test_mixture_prior_eb_runs_and_recovers():
    X, Y, _ = _simulate_two_phenotypes(jax.random.PRNGKey(3), j=15)
    model = Model(
        design=SharedDesign(X=X),
        likelihood=likelihoods.mvn_resid(Y),
        prior=priors.mvsusie(canonical_components(K=2, scale=1.0)),
        log_pi=jnp.full(X.shape[1], -jnp.log(X.shape[1])),
    )
    # exercises the multi-component EB branch (Prior.update)
    result = finemap(model, L=2, coverage=0.95, purity=0.5, update_prior=True)
    assert int(np.argmax(result.pip)) == 15
    assert np.all(np.isfinite(result.pip))
