"""The composite Cox proportional-hazards likelihood.

  * ``CoxPH.log_density`` (cumlogsumexp risk sets after a one-time sort) matches a
    brute-force O(N^2) Breslow partial log-likelihood (distinct times).
  * Preprocessing is data-derived and eta-independent (a gradient exists and is finite).
  * ``expected_log_density`` applies the diagonal delta-method correction (matches a
    from-scratch reference), is a no-op at var=0, and its correction is disconnected from
    the mode-finding gradient (the decoupling documented in
    ``model.AbstractLikelihood.expected_log_density``).
  * SuSiE with the Cox likelihood recovers a planted survival association.
"""

import jax
import jax.numpy as jnp
import numpy as np

from lasusie import finemap, laplace, likelihoods, priors
from lasusie.design import SharedDesign
from lasusie.model import Model


def _brute_force_cox(times, events, eta):
    times = np.asarray(times)
    events = np.asarray(events)
    eta = np.asarray(eta)
    ll = 0.0
    for i in np.where(events == 1)[0]:
        risk = np.where(times >= times[i])[0]
        m = eta[risk].max()
        ll += eta[i] - (m + np.log(np.sum(np.exp(eta[risk] - m))))
    return ll


def test_cox_log_density_matches_bruteforce():
    key = jax.random.PRNGKey(0)
    N = 40
    times = jax.random.uniform(key, (N,), minval=0.1, maxval=10.0)  # distinct a.s.
    events = jax.random.bernoulli(jax.random.PRNGKey(1), 0.7, (N,)).astype(float)
    eta = jax.random.normal(jax.random.PRNGKey(2), (N,))

    lik = likelihoods.cox(times, events)
    got = float(lik.log_density(eta))
    expected = _brute_force_cox(times, events, eta)
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_cox_gradient_is_finite():
    key = jax.random.PRNGKey(3)
    N = 30
    times = jax.random.uniform(key, (N,), minval=0.1, maxval=5.0)
    events = jnp.ones(N)
    lik = likelihoods.cox(times, events)
    g = jax.grad(lik.log_density)(jax.random.normal(jax.random.PRNGKey(4), (N,)))
    assert np.all(np.isfinite(np.asarray(g)))


def test_expected_log_density_matches_delta_reference():
    key = jax.random.PRNGKey(6)
    N = 25
    times = jax.random.uniform(key, (N,), minval=0.1, maxval=8.0)
    events = jax.random.bernoulli(jax.random.PRNGKey(7), 0.7, (N,)).astype(float)
    mean = jax.random.normal(jax.random.PRNGKey(8), (N,)) * 0.3
    var = jax.nn.softplus(jax.random.normal(jax.random.PRNGKey(9), (N,)))

    lik = likelihoods.cox(times, events)
    got = float(lik.expected_log_density(mean, var))

    H = jax.hessian(lik.log_density)(mean)
    expected = float(lik.log_density(mean)) + 0.5 * float(jnp.sum(jnp.diagonal(H) * var))
    np.testing.assert_allclose(got, expected, rtol=1e-8)


def test_hessian_diagonal_matches_dense_reference():
    # The O(N) closed-form diagonal must equal jnp.diagonal(jax.hessian(...)) exactly,
    # across censoring and a warm eta far from zero (where saturation could bite).
    key = jax.random.PRNGKey(20)
    N = 35
    times = jax.random.uniform(key, (N,), minval=0.1, maxval=9.0)
    events = jax.random.bernoulli(jax.random.PRNGKey(21), 0.6, (N,)).astype(float)
    eta = jax.random.normal(jax.random.PRNGKey(22), (N,)) * 1.5

    lik = likelihoods.cox(times, events)
    got = lik.hessian_diagonal(eta)
    expected = jnp.diagonal(jax.hessian(lik.log_density)(eta))
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-6)


def test_expected_log_density_at_zero_var_is_plain_log_density():
    key = jax.random.PRNGKey(10)
    N = 20
    times = jax.random.uniform(key, (N,), minval=0.1, maxval=6.0)
    events = jnp.ones(N)
    eta = jax.random.normal(jax.random.PRNGKey(11), (N,))
    lik = likelihoods.cox(times, events)

    plain = lik.log_density(eta)
    quad = lik.expected_log_density(eta, jnp.zeros(N))
    np.testing.assert_allclose(quad, plain, atol=1e-9)


def test_correction_does_not_perturb_mode_or_precision():
    # Same warm start / offset mean, only var differs -> Laplace fit's mode and precision
    # (found via the *plug-in* objective) must be identical; only log_scale should differ.
    key = jax.random.PRNGKey(12)
    N, K = 40, 1
    times = jax.random.uniform(key, (N,), minval=0.1, maxval=8.0)
    events = jax.random.bernoulli(jax.random.PRNGKey(13), 0.7, (N,)).astype(float)
    x_m = jax.random.normal(jax.random.PRNGKey(14), (N,))
    offset_mean = jax.random.normal(jax.random.PRNGKey(15), (N,)) * 0.2
    lik = likelihoods.cox(times, events)

    def make_g(var):
        def g(beta):
            eta = offset_mean + x_m * beta[0]
            return lik.expected_log_density(eta, var)

        return g

    pot_zero = laplace.fit(make_g(jnp.zeros(N)), jnp.zeros(K))
    pot_var = laplace.fit(make_g(jnp.full(N, 0.5)), jnp.zeros(K))

    np.testing.assert_allclose(pot_zero.mode, pot_var.mode, atol=1e-8)
    np.testing.assert_allclose(pot_zero.precision, pot_var.precision, atol=1e-8)
    assert not np.allclose(pot_zero.log_scale, pot_var.log_scale, atol=1e-6)


def test_cox_recovers_survival_association():
    key = jax.random.PRNGKey(5)
    kx, kt = jax.random.split(key)
    N, M, j = 600, 25, 6
    X = jax.random.normal(kx, (N, M))
    X = (X - X.mean(0)) / X.std(0)
    eta_true = X[:, j] * 1.5
    # exponential survival times with hazard exp(eta): T = -log(U)/exp(eta)
    U = jax.random.uniform(kt, (N,), minval=1e-6, maxval=1.0)
    times = -jnp.log(U) / jnp.exp(eta_true)
    events = jnp.ones(N)

    model = Model(
        design=SharedDesign(X=X),
        likelihood=likelihoods.cox(times, events),
        prior=priors.susie(1.0),
        log_pi=jnp.full(M, -jnp.log(M)),
    )
    result = finemap(model, L=2, coverage=0.95, purity=0.5)
    assert int(np.argmax(result.pip)) == j
    assert result.pip[j] > 0.9
