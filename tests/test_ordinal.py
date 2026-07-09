"""Ordinal (cumulative-link) likelihood with EB-estimated cutpoints.

  * The per-category probabilities are a proper distribution (sum to 1) and match the
    direct CDF-difference definition, for both the logit and probit links.
  * ``expected_log_density`` at zero offset variance reduces to ``log_density``.
  * The EB cutpoint M-step recovers the generating cutpoints from data.
  * Fine-mapping recovers a strong causal variant with an ordinal outcome, and the
    cutpoints are EB-updated toward the truth.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from lasusie import Model, finemap, likelihoods, priors
from lasusie.design import SharedDesign
from lasusie.likelihoods import _cutpoints


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _theta(cuts):
    return {
        "cut0": jnp.asarray(cuts[0]),
        "log_deltas": jnp.log(jnp.asarray(np.diff(cuts))),
    }


def _category_probs(lik, theta, eta0, C):
    eta = jnp.array([[eta0]])
    return np.array(
        [float(lik.factor(jnp.array([[c]], float), eta, theta).reshape(())) for c in range(C)]
    )


def test_ordinal_probabilities_match_cdf_and_normalise():
    C = 4
    cuts = np.array([-0.8, 0.2, 1.1])
    eta0 = 0.3
    theta = _theta(cuts)

    for link, cdf in [("logit", _sigmoid), ("probit", stats.norm.cdf)]:
        lik = likelihoods.ordinal_logit(np.array([0]), C, cutpoints=cuts)
        if link == "probit":
            lik = likelihoods.ordinal_probit(np.array([0]), C, cutpoints=cuts)
        probs = np.exp(_category_probs(lik, theta, eta0, C))
        ref = np.array(
            [
                (cdf(cuts[c] - eta0) if c < C - 1 else 1.0)
                - (cdf(cuts[c - 1] - eta0) if c > 0 else 0.0)
                for c in range(C)
            ]
        )
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12, err_msg=link)
        np.testing.assert_allclose(probs, ref, atol=1e-12, err_msg=link)


def test_ordinal_expected_at_zero_var_equals_log_density():
    C = 4
    y = jnp.asarray(np.random.default_rng(0).integers(0, C, 40).reshape(-1, 1), float)
    eta = jax.random.normal(jax.random.PRNGKey(1), (40, 1))
    for lik in (likelihoods.ordinal_logit(y, C), likelihoods.ordinal_probit(y, C)):
        direct = lik.log_density(eta)
        quad = lik.expected_log_density(eta, jnp.zeros_like(eta))
        np.testing.assert_allclose(quad, direct, rtol=1e-9, atol=1e-8)


def test_ordinal_eb_recovers_cutpoints():
    C = 4
    true = np.array([-1.0, 0.5, 1.5])
    N = 6000
    u = np.random.default_rng(2).random(N)
    thr = _sigmoid(true)  # P(Y <= c) at eta=0
    ycat = np.array([np.searchsorted(thr, uu) for uu in u])

    lik = likelihoods.ordinal_logit(ycat.reshape(-1, 1), C)  # default (wrong) init
    zeros = jnp.zeros((N, 1))
    for _ in range(8):
        lik = lik.updated(zeros, zeros)  # EB M-step at eta = 0
    got = np.asarray(_cutpoints(lik.theta))
    np.testing.assert_allclose(got, true, atol=0.1)


def test_ordinal_finemap_recovers_causal():
    X = jax.random.normal(jax.random.PRNGKey(7), (400, 30))
    X = X - X.mean(0)
    N, M = X.shape
    C, causal = 4, 13
    eta = np.asarray(X[:, causal] * 1.5)
    true = np.array([-1.0, 0.3, 1.4])
    thr = _sigmoid(true[None, :] - eta[:, None])  # P(Y <= c | eta)
    u = np.random.default_rng(7).random(N)
    ycat = (u[:, None] > thr).sum(1)  # category 0..C-1

    model = Model(
        design=SharedDesign(X=X),
        likelihood=likelihoods.ordinal_logit(ycat.reshape(-1, 1), C),
        prior=priors.susie(1.0),
        log_pi=jnp.full(M, -jnp.log(M)),
    )
    res = finemap(model, L=3, coverage=0.95, purity=0.5)
    assert not np.isnan(res.pip).any()
    assert res.pip[causal] > 0.9
    # cutpoints EB-updated toward the generating values
    np.testing.assert_allclose(np.asarray(_cutpoints(res.posterior.likelihood.theta)), true, atol=0.3)
