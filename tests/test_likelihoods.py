"""Correctness of the concrete pointwise likelihoods and their variance propagation.

Three groups of checks:
  * the variance-propagating quadrature (``expected_log_density``) matches the closed-form
    Gaussian expected log-density ("ER^2") and, at zero offset variance, reduces to
    ``log_density`` -- the invariant every pointwise likelihood must satisfy for the SER;
  * each of the common likelihoods has its per-observation ``factor`` matched against an
    independent SciPy reference density; and
  * each likelihood fine-maps a strong planted causal end-to-end.

The count / robust / continuous / survival factors are all single-phenotype (K=1), so
``eta``/``y`` carry a trailing singleton axis, matching the ``(N, 1)`` convention the
IBSS loop feeds them. The multi-phenotype ``mvn_resid`` likelihood is exercised here too,
via the SER precision it induces (a dense K x K block from the residual covariance).
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from lasusie import Model, finemap, likelihoods, priors, shared_covariates
from lasusie.design import SharedDesign
from lasusie.laplace import fit


def _zero_var_matches_log_density(lik, eta):
    """expected_log_density(eta, 0) == log_density(eta) for a pointwise likelihood."""
    direct = lik.log_density(eta)
    quad = lik.expected_log_density(eta, jnp.zeros_like(eta))
    np.testing.assert_allclose(quad, direct, rtol=1e-9, atol=1e-8)


# --- Variance-propagating quadrature (the SER's expected_log_density) -----------------


def test_quadrature_matches_gaussian_er2():
    key = jax.random.PRNGKey(0)
    ky, km, kv = jax.random.split(key, 3)
    N = 50
    y = jax.random.normal(ky, (N,))
    mean = jax.random.normal(km, (N,))
    var = jax.nn.softplus(jax.random.normal(kv, (N,))) + 0.1  # positive
    sigma2 = 0.7

    lik = likelihoods.gaussian(y, sigma2=sigma2)
    got = lik.expected_log_density(mean, var)

    exp = jnp.sum(
        -0.5 * jnp.log(2 * jnp.pi * sigma2) - 0.5 * ((y - mean) ** 2 + var) / sigma2
    )
    np.testing.assert_allclose(got, exp, rtol=1e-10, atol=1e-9)


def test_expected_at_zero_var_equals_log_density():
    key = jax.random.PRNGKey(1)
    y = jax.random.bernoulli(key, 0.5, (30,)).astype(float)
    eta = jax.random.normal(jax.random.PRNGKey(2), (30,))
    lik = likelihoods.bernoulli_logit(y)

    direct = lik.log_density(eta)
    quad = lik.expected_log_density(eta, jnp.zeros_like(eta))
    np.testing.assert_allclose(quad, direct, rtol=1e-10, atol=1e-9)


def test_mvn_resid_ser_precision_is_dense_from_resid_cov():
    # The SER precision for a shared-X, K-variate Gaussian likelihood equals
    # (sum_n X[n, m]^2) * Sigma_e^{-1} -- a dense K x K matrix whose off-diagonal comes
    # from the residual covariance (coupling through the likelihood, not only the prior).
    key = jax.random.PRNGKey(0)
    N, K, m = 60, 2, 3
    M = 10
    X = jax.random.normal(key, (N, M))
    Sigma_e = jnp.array([[1.0, 0.5], [0.5, 1.3]])
    Y = jax.random.normal(jax.random.PRNGKey(1), (N, K))
    lik = likelihoods.mvn_resid(Y, resid_cov=Sigma_e)

    # single-effect objective at variant m against a zero offset
    def g(beta):
        eta = jnp.outer(X[:, m], beta)  # (N, K)
        return lik.expected_log_density(eta, jnp.zeros((N, K)))

    pot = fit(g, jnp.zeros(K))
    expected = jnp.sum(X[:, m] ** 2) * jnp.linalg.inv(Sigma_e)

    np.testing.assert_allclose(pot.precision, expected, atol=1e-6)
    # genuinely dense: off-diagonal is non-trivial
    assert abs(float(pot.precision[0, 1])) > 1e-3


# --- Per-likelihood factor correctness against SciPy references -----------------------


def test_neg_binomial_matches_scipy():
    key = jax.random.PRNGKey(0)
    eta = jax.random.normal(key, (40, 1))
    mu = np.exp(np.asarray(eta))
    r = 2.5
    y = jnp.asarray(np.random.default_rng(0).poisson(mu)).astype(float)

    lik = likelihoods.neg_binomial_log(y, r=r)
    got = lik.factor(y, eta, {"r": jnp.asarray(r)})
    ref = stats.nbinom.logpmf(np.asarray(y), n=r, p=r / (r + mu))
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-9, atol=1e-8)
    _zero_var_matches_log_density(lik, eta)


def test_student_t_matches_scipy_and_gaussian_limit():
    key = jax.random.PRNGKey(1)
    y = jax.random.normal(key, (40, 1))
    eta = jax.random.normal(jax.random.PRNGKey(2), (40, 1))
    nu, sigma2 = 5.0, 0.7

    lik = likelihoods.student_t(y, nu=nu, sigma2=sigma2)
    got = lik.factor(y, eta, {"nu": jnp.asarray(nu), "sigma2": jnp.asarray(sigma2)})
    ref = stats.t.logpdf(np.asarray(y), df=nu, loc=np.asarray(eta), scale=np.sqrt(sigma2))
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-9, atol=1e-8)
    _zero_var_matches_log_density(lik, eta)

    # Large nu -> Gaussian.
    lik_big = likelihoods.student_t(y, nu=1e6, sigma2=sigma2)
    gauss = likelihoods.gaussian(y, sigma2=sigma2)
    np.testing.assert_allclose(
        lik_big.log_density(eta), gauss.log_density(eta), rtol=1e-4, atol=1e-4
    )


def test_gamma_matches_scipy():
    eta = jax.random.normal(jax.random.PRNGKey(3), (40, 1))
    mu = np.exp(np.asarray(eta))
    k = 2.0
    y = jnp.asarray(np.random.default_rng(1).gamma(shape=k, scale=mu / k)).astype(float)

    lik = likelihoods.gamma_log(y, shape=k)
    got = lik.factor(y, eta, {"shape": jnp.asarray(k)})
    ref = stats.gamma.logpdf(np.asarray(y), a=k, scale=mu / k)  # scale = mu / k
    np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-9, atol=1e-8)
    _zero_var_matches_log_density(lik, eta)


def test_beta_binomial_matches_scipy():
    eta = jax.random.normal(jax.random.PRNGKey(4), (30, 1))
    p = np.asarray(jax.nn.sigmoid(eta))
    s = 8.0
    n = 20
    y = jnp.asarray(np.random.default_rng(2).binomial(n, p)).astype(float)

    lik = likelihoods.beta_binomial_logit(y.reshape(-1), n, s=s)
    got = lik.log_density(eta)
    ref = stats.betabinom.logpmf(np.asarray(y), n=n, a=p * s, b=(1 - p) * s).sum()
    np.testing.assert_allclose(float(got), float(ref), rtol=1e-9, atol=1e-7)
    _zero_var_matches_log_density(lik, eta)


def _aft_reference(dist_kwargs_fn, times, events, eta):
    """Sum of scipy logpdf (events) / logsf (censored) for an AFT reference."""
    t = np.asarray(times)
    d = np.asarray(events)
    dist = dist_kwargs_fn(np.asarray(eta).reshape(-1))
    return np.sum(np.where(d == 1, dist.logpdf(t), dist.logsf(t)))


def test_aft_models_match_scipy():
    rng = np.random.default_rng(5)
    n = 30
    times = jnp.asarray(rng.uniform(0.2, 5.0, n))
    events = jnp.asarray((rng.random(n) > 0.3).astype(float))
    eta = jax.random.normal(jax.random.PRNGKey(6), (n, 1))
    scale = 0.8
    eta_flat = np.asarray(eta).reshape(-1)

    cases = {
        "lognormal": (
            likelihoods.aft_lognormal(times, events, scale=scale),
            lambda e: stats.lognorm(s=scale, scale=np.exp(e)),
        ),
        # Weibull AFT: shape c = 1/scale, scale = exp(eta).
        "weibull": (
            likelihoods.aft_weibull(times, events, scale=scale),
            lambda e: stats.weibull_min(c=1.0 / scale, scale=np.exp(e)),
        ),
        # Log-logistic AFT: scipy's Fisk with c = 1/scale, scale = exp(eta).
        "loglogistic": (
            likelihoods.aft_loglogistic(times, events, scale=scale),
            lambda e: stats.fisk(c=1.0 / scale, scale=np.exp(e)),
        ),
    }
    for name, (lik, dist_fn) in cases.items():
        got = float(lik.log_density(eta))
        ref = _aft_reference(dist_fn, times, events, eta)
        np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-6, err_msg=name)
        _zero_var_matches_log_density(lik, eta)


def test_aft_lognormal_all_events_is_censoring_free_lognormal():
    """With no censoring, log-normal AFT equals a Gaussian likelihood on log-times."""
    rng = np.random.default_rng(7)
    times = jnp.asarray(rng.uniform(0.5, 4.0, 25))
    eta = jax.random.normal(jax.random.PRNGKey(8), (25, 1))
    scale = 0.9

    aft = likelihoods.aft_lognormal(times, events=None, scale=scale)
    # log N(log t; eta, scale^2) differs from the AFT density only by the -log t Jacobian.
    gauss = likelihoods.gaussian(jnp.log(times).reshape(-1, 1), sigma2=scale**2)
    jac = -jnp.sum(jnp.log(times))
    np.testing.assert_allclose(
        float(aft.log_density(eta)), float(gauss.log_density(eta) + jac), rtol=1e-9, atol=1e-8
    )


# --- End-to-end fine-mapping: each likelihood recovers a strong causal variant --------


def _genotypes(key, N=300, M=30):
    """Correlated-ish standardised genotypes with one causal column."""
    X = jax.random.normal(key, (N, M))
    return X - X.mean(0)


def _finemap_recovers(model, causal, L=2):
    result = finemap(model, L=L, coverage=0.95, purity=0.5)
    kept = [cs for cs in result.credible_sets if cs.kept]
    in_cs = any(causal in cs.variants for cs in kept)
    top = int(np.argmax(np.asarray(result.pip)))
    return in_cs or top == causal


def _intercept(N):
    return shared_covariates(jnp.ones((N, 1)), K=1, add_intercept=False)


def test_finemap_neg_binomial():
    key = jax.random.PRNGKey(10)
    X = _genotypes(key)
    N, M = X.shape
    causal, b0 = 7, 1.5
    eta = b0 + X[:, causal] * 1.2
    mu = np.exp(np.asarray(eta))
    r = 3.0
    y = np.random.default_rng(10).negative_binomial(r, r / (r + mu)).astype(float)

    model = Model(
        design=SharedDesign(X=X),
        likelihood=likelihoods.neg_binomial_log(jnp.asarray(y).reshape(-1, 1), r=r),
        prior=priors.susie(1.0),
        log_pi=jnp.full(M, -jnp.log(M)),
        covariates=_intercept(N),
    )
    assert _finemap_recovers(model, causal)


def test_finemap_aft_weibull():
    key = jax.random.PRNGKey(11)
    X = _genotypes(key)
    N, M = X.shape
    causal, scale = 15, 0.6
    eta = np.asarray(X[:, causal] * 1.4)
    rng = np.random.default_rng(11)
    # log T = eta + scale * Gumbel_min ; censor the longest ~25%.
    w = np.log(rng.exponential(size=N))  # standard min-Gumbel
    times = np.exp(eta + scale * w)
    cutoff = np.quantile(times, 0.75)
    events = (times <= cutoff).astype(float)
    times = np.minimum(times, cutoff)

    model = Model(
        design=SharedDesign(X=X),
        likelihood=likelihoods.aft_weibull(times, events, scale=scale),
        prior=priors.susie(1.0),
        log_pi=jnp.full(M, -jnp.log(M)),
    )
    assert _finemap_recovers(model, causal)


def test_finemap_beta_binomial():
    key = jax.random.PRNGKey(12)
    X = _genotypes(key)
    N, M = X.shape
    causal, s, n = 22, 15.0, 40
    eta = np.asarray(X[:, causal] * 1.6)
    p = 1.0 / (1.0 + np.exp(-eta))
    rng = np.random.default_rng(12)
    a, b = p * s, (1 - p) * s
    y = rng.binomial(n, rng.beta(a, b)).astype(float)

    model = Model(
        design=SharedDesign(X=X),
        likelihood=likelihoods.beta_binomial_logit(y, n, s=s),
        prior=priors.susie(1.0),
        log_pi=jnp.full(M, -jnp.log(M)),
    )
    assert _finemap_recovers(model, causal)
