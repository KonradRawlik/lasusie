"""The propagate_variance switch: likelihood-agnostic, selects log_density vs
expected_log_density in the SER objective (and in the reported ELBO).

  * At zero offset variance, the two must agree exactly (no correction to ignore) --
    for both a pointwise (Gaussian) and a composite (Cox) likelihood.
  * At nonzero offset variance, they must genuinely differ -- confirming the flag isn't
    a no-op -- while both still recover the correct causal variant.
  * Turning it off must skip the expensive Hessian computation from the Cox delta method
    (measured to confirm the intended cost saving is real, not just documented).
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from lasusie import finemap, likelihoods, priors
from lasusie.design import SharedDesign
from lasusie.ibss import single_effect_regression
from lasusie.model import Model


def test_zero_offset_variance_flag_is_a_noop_gaussian():
    key = jax.random.PRNGKey(0)
    N, M = 100, 10
    X = jax.random.normal(key, (N, M))
    y = jax.random.normal(jax.random.PRNGKey(1), (N, 1))
    design = SharedDesign(X=X)
    lik = likelihoods.gaussian(y, sigma2=1.0)
    prior = priors.susie(1.0)
    log_pi = jnp.full(M, -jnp.log(M))
    zero_var = jnp.zeros((N, 1))

    a_true, post_true, _, _ = single_effect_regression(
        design, lik, prior, log_pi, jnp.zeros((N, 1)), zero_var, jnp.zeros((M, 1)), True
    )
    a_false, post_false, _, _ = single_effect_regression(
        design, lik, prior, log_pi, jnp.zeros((N, 1)), zero_var, jnp.zeros((M, 1)), False
    )
    np.testing.assert_allclose(a_true, a_false, atol=1e-8)
    np.testing.assert_allclose(post_true.mixture_mean, post_false.mixture_mean, atol=1e-8)


def test_zero_offset_variance_is_noop_regardless_of_gaussian_var_value():
    # For a Gaussian likelihood, f''(eta) = -1/sigma2 is *constant* (independent of eta),
    # so the exact correction E[f] = f(mean) - 0.5*var/sigma2 is the same additive shift
    # for every candidate variant m -- it cancels in the softmax over m. propagate_variance
    # therefore provably cannot change alpha/PIPs for a Gaussian likelihood (only the
    # absolute ELBO value); this documents that invariant rather than fighting it.
    key = jax.random.PRNGKey(2)
    N, M = 100, 10
    X = jax.random.normal(key, (N, M))
    y = jax.random.normal(jax.random.PRNGKey(3), (N, 1))
    design = SharedDesign(X=X)
    lik = likelihoods.gaussian(y, sigma2=1.0)
    prior = priors.susie(1.0)
    log_pi = jnp.full(M, -jnp.log(M))
    nonzero_var = jnp.full((N, 1), 0.8)

    a_true, post_true, _, _ = single_effect_regression(
        design, lik, prior, log_pi, jnp.zeros((N, 1)), nonzero_var, jnp.zeros((M, 1)), True
    )
    a_false, post_false, _, _ = single_effect_regression(
        design, lik, prior, log_pi, jnp.zeros((N, 1)), nonzero_var, jnp.zeros((M, 1)), False
    )
    np.testing.assert_allclose(a_true, a_false, atol=1e-8)


def test_nonzero_offset_variance_flag_changes_result_logistic():
    # Unlike Gaussian, logistic's curvature f''(eta) = -sigmoid(eta)(1-sigmoid(eta)) varies
    # with eta, so it varies with beta (and hence with which variant m is tested) -- the
    # correction is genuinely beta-dependent here and can shift the variant ranking.
    key = jax.random.PRNGKey(20)
    N, M = 200, 10
    X = jax.random.normal(key, (N, M))
    y = jax.random.bernoulli(jax.random.PRNGKey(21), 0.5, (N, 1)).astype(float)
    design = SharedDesign(X=X)
    lik = likelihoods.bernoulli_logit(y)
    prior = priors.susie(1.0)
    log_pi = jnp.full(M, -jnp.log(M))
    nonzero_var = jnp.full((N, 1), 1.5)

    a_true, _, _, _ = single_effect_regression(
        design, lik, prior, log_pi, jnp.zeros((N, 1)), nonzero_var, jnp.zeros((M, 1)), True
    )
    a_false, _, _, _ = single_effect_regression(
        design, lik, prior, log_pi, jnp.zeros((N, 1)), nonzero_var, jnp.zeros((M, 1)), False
    )
    assert not np.allclose(a_true, a_false, atol=1e-6)


def test_gaussian_recovers_causal_both_orders():
    key = jax.random.PRNGKey(4)
    kx, kn = jax.random.split(key)
    N, M, j = 400, 30, 11
    X = jax.random.normal(kx, (N, M))
    X = (X - X.mean(0)) / X.std(0)
    y = X[:, j] * 1.3 + jax.random.normal(kn, (N,))

    for order in (True, False):
        model = Model(
            design=SharedDesign(X=X),
            likelihood=likelihoods.gaussian(y.reshape(-1, 1), sigma2=1.0),
            prior=priors.susie(1.0),
            log_pi=jnp.full(M, -jnp.log(M)),
        )
        result = finemap(model, L=2, coverage=0.95, purity=0.5, propagate_variance=order)
        assert int(np.argmax(result.pip)) == j, f"failed for propagate_variance={order}"


def _cox_setup(key, N=300, M=15, j=4):
    kx, kt = jax.random.split(key)
    X = jax.random.normal(kx, (N, M))
    X = (X - X.mean(0)) / X.std(0)
    eta_true = X[:, j] * 1.4
    U = jax.random.uniform(kt, (N,), minval=1e-6, maxval=1.0)
    times = -jnp.log(U) / jnp.exp(eta_true)
    events = jnp.ones(N)
    return X, times, events, j


def test_cox_zero_var_flag_is_noop():
    X, times, events, j = _cox_setup(jax.random.PRNGKey(5))
    design = SharedDesign(X=X)
    lik = likelihoods.cox(times, events)
    prior = priors.susie(1.0)
    log_pi = jnp.full(X.shape[1], -jnp.log(X.shape[1]))
    zero_var = jnp.zeros((X.shape[0], 1))

    a_true, _, _, _ = single_effect_regression(
        design, lik, prior, log_pi, jnp.zeros((X.shape[0], 1)), zero_var,
        jnp.zeros((X.shape[1], 1)), True,
    )
    a_false, _, _, _ = single_effect_regression(
        design, lik, prior, log_pi, jnp.zeros((X.shape[0], 1)), zero_var,
        jnp.zeros((X.shape[1], 1)), False,
    )
    np.testing.assert_allclose(a_true, a_false, atol=1e-6)


def test_cox_recovers_causal_both_orders_and_variance_path_is_cheap():
    # L=1 here means offset_var is always exactly zero (no other effect contributes
    # uncertainty), so this specifically checks the "wasted computation at var=0" case --
    # it does NOT show that the correction is a no-op for Cox in general (see the next
    # test, which uses L=2 / nonzero offset variance to check that).
    X, times, events, j = _cox_setup(jax.random.PRNGKey(6), N=400, M=20)
    timings = {}
    for order in (True, False):
        model = Model(
            design=SharedDesign(X=X),
            likelihood=likelihoods.cox(times, events),
            prior=priors.susie(1.0),
            log_pi=jnp.full(X.shape[1], -jnp.log(X.shape[1])),
        )
        start = time.perf_counter()
        result = finemap(model, L=1, coverage=0.95, purity=0.5, propagate_variance=order)
        timings[order] = time.perf_counter() - start
        assert int(np.argmax(result.pip)) == j, f"failed for propagate_variance={order}"

    # Since CoxPH.hessian_diagonal computes the delta correction in O(N) (risk-set
    # cumulative sums) instead of materialising the O(N^2) dense Hessian per SER
    # evaluation, propagating variance is now at most marginally more expensive than the
    # zeroth-order path -- not the multiple-times-slower it used to be. A regression back
    # to the dense Hessian would blow this ratio up well past the bound.
    assert timings[True] < timings[False] * 2.0


def test_cox_nonzero_variance_shifts_alpha_when_evidence_is_close():
    # Unlike Gaussian, Cox's curvature H_nn(eta) = d^2 log PL / d eta_n^2 is NOT constant
    # in eta, so the delta correction genuinely varies across candidate variants (mode/
    # precision stay decoupled from var -- see test_correction_does_not_perturb_mode_or_
    # precision in test_cox.py -- but log_scale, and hence alpha, is not). With a
    # strong/unambiguous effect the shift is real but gets swamped by an already-huge
    # evidence gap (softmax saturation hides it); with a weak effect, where variants have
    # comparable evidence, the shift is visible directly in alpha.
    key = jax.random.PRNGKey(30)
    kx, kt = jax.random.split(key)
    N, M, j = 200, 15, 4
    X = jax.random.normal(kx, (N, M))
    X = (X - X.mean(0)) / X.std(0)
    eta_true = X[:, j] * 0.25  # weak effect -> variants have comparable evidence
    U = jax.random.uniform(kt, (N,), minval=1e-6, maxval=1.0)
    times = -jnp.log(U) / jnp.exp(eta_true)
    events = jnp.ones(N)

    design = SharedDesign(X=X)
    lik = likelihoods.cox(times, events)
    prior = priors.susie(1.0)
    log_pi = jnp.full(M, -jnp.log(M))
    nonzero_var = jnp.full((N, 1), 0.8)

    a_true, _, _, _ = single_effect_regression(
        design, lik, prior, log_pi, jnp.zeros((N, 1)), nonzero_var, jnp.zeros((M, 1)), True
    )
    a_false, _, _, _ = single_effect_regression(
        design, lik, prior, log_pi, jnp.zeros((N, 1)), nonzero_var, jnp.zeros((M, 1)), False
    )
    assert not np.allclose(a_true, a_false, atol=1e-4)
