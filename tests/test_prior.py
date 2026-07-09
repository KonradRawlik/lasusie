"""The mixture prior's closed-form ``combine`` matches known SER formulas.

  * K=1 single Gaussian == susieR single_effect_regression (log-BF + posterior).
  * a mixture of identical components == the single component (moments + evidence).
  * K>1 single dense component == the sushie ``_compute_posterior`` direct-inverse form.
"""

import jax.numpy as jnp
import numpy as np

from lasusie import priors
from lasusie.laplace import GaussianPotential
from lasusie.model import Prior


def potential(precision, mode, log_scale=0.0):
    precision = jnp.asarray(precision, dtype=float)
    mode = jnp.asarray(mode, dtype=float)
    return GaussianPotential(
        log_scale=jnp.asarray(log_scale), mode=mode, precision=precision
    )


def null_log_evidence(pot):
    """log-likelihood at beta=0 = log_scale - 0.5 mode^T P mode."""
    return pot.log_scale - 0.5 * pot.mode @ pot.precision @ pot.mode


def test_k1_matches_susieR_formula():
    s2, b_hat, sigma0_sq = 0.3, 1.7, 2.5
    pot = potential([[1.0 / s2]], [b_hat])
    prior = priors.susie(sigma0_sq)

    post, log_marginal = prior.combine(pot)

    # susieR closed forms
    exp_post_var = 1.0 / (1.0 / s2 + 1.0 / sigma0_sq)
    exp_post_mean = (sigma0_sq / (s2 + sigma0_sq)) * b_hat
    exp_lbf = 0.5 * np.log(s2 / (s2 + sigma0_sq)) + 0.5 * b_hat**2 * sigma0_sq / (
        s2 * (s2 + sigma0_sq)
    )

    np.testing.assert_allclose(post.resp, [1.0], atol=1e-10)
    np.testing.assert_allclose(post.mean[0, 0], exp_post_mean, atol=1e-10)
    np.testing.assert_allclose(post.covar[0, 0, 0], exp_post_var, atol=1e-10)
    lbf = log_marginal - null_log_evidence(pot)
    np.testing.assert_allclose(lbf, exp_lbf, atol=1e-10)


def test_mixture_of_identical_equals_single():
    s2, b_hat, sigma0_sq = 0.5, -0.9, 1.3
    pot = potential([[1.0 / s2]], [b_hat])

    single = priors.susie(sigma0_sq)
    duplicated = Prior(
        log_weights=jnp.zeros(2),
        covariances=jnp.array([[[sigma0_sq]], [[sigma0_sq]]]),
    )

    p1, z1 = single.combine(pot)
    p2, z2 = duplicated.combine(pot)

    np.testing.assert_allclose(z1, z2, atol=1e-10)
    np.testing.assert_allclose(p1.mixture_mean, p2.mixture_mean, atol=1e-10)
    np.testing.assert_allclose(
        p1.mixture_second_moment, p2.mixture_second_moment, atol=1e-10
    )
    np.testing.assert_allclose(p2.resp, [0.5, 0.5], atol=1e-10)


def test_k2_matches_sushie_direct_inverse():
    K = 2
    P = jnp.array([[4.0, 0.5], [0.5, 3.0]])  # likelihood precision (inv_shat2)
    b_hat = jnp.array([0.8, -1.2])
    Sigma = jnp.array([[1.5, 0.3], [0.3, 0.9]])  # dense prior covariance
    pot = potential(P, b_hat)
    prior = priors.sushie(Sigma)

    post, _ = prior.combine(pot)

    # sushie _compute_posterior direct-inverse form
    exp_covar = jnp.linalg.inv(P + jnp.linalg.inv(Sigma))
    exp_mean = exp_covar @ (P @ b_hat)

    np.testing.assert_allclose(post.mean[0], exp_mean, atol=1e-9)
    np.testing.assert_allclose(post.covar[0], exp_covar, atol=1e-9)


def test_update_recovers_weighted_second_moment():
    # single component: EB update == alpha-weighted average second moment (sushie).
    K = 2
    P = jnp.array([[4.0, 0.5], [0.5, 3.0]])
    prior = priors.sushie(jnp.array([[1.5, 0.3], [0.3, 0.9]]))

    modes = jnp.array([[0.8, -1.2], [0.1, 0.4], [-0.5, 0.3]])
    import jax

    pots = jax.vmap(lambda m: potential(P, m))(modes)
    post, _ = jax.vmap(prior.combine)(pots)  # batched over M=3
    alpha = jnp.array([0.6, 0.3, 0.1])

    new_prior = prior.update(alpha, post)

    # single component => Sigma = sum_m alpha_m E[bb^T|m] / sum_m alpha_m
    second = post.covar[:, 0] + jnp.einsum("mk,ml->mkl", post.mean[:, 0], post.mean[:, 0])
    exp_Sigma = jnp.einsum("m,mkl->kl", alpha, second) / alpha.sum()
    np.testing.assert_allclose(new_prior.covariances[0], exp_Sigma, atol=1e-9)
