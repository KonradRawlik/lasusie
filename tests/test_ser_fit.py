"""The two single-effect-regression fit modes: conjugate (default) and joint (MAP).

  * For a Gaussian likelihood the joint per-component fit is exact, so it must reproduce
    the conjugate one-fit-plus-analytic-combine path to floating precision.
  * A count/exponential-family likelihood has a divergent per-variant MLE for null
    variants; both the numerical guards in ``laplace.fit`` (conjugate path) and the joint
    MAP fit must keep fine-mapping finite and recover a strong causal variant even at a
    generous ``L``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from lasusie import Model, finemap, likelihoods, priors, shared_covariates
from lasusie.design import SharedDesign


def _genotypes(key, N=300, M=30):
    X = jax.random.normal(key, (N, M))
    return X - X.mean(0)


def test_joint_equals_conjugate_for_gaussian():
    X = _genotypes(jax.random.PRNGKey(0))
    N, M = X.shape
    causal = 9
    y = 0.5 + X[:, causal] * 1.3 + 0.4 * jax.random.normal(jax.random.PRNGKey(1), (N,))
    y = jnp.asarray(y).reshape(-1, 1)

    def fit(ser):
        model = Model(
            design=SharedDesign(X=X),
            likelihood=likelihoods.gaussian(y, sigma2=0.16),
            prior=priors.susie(1.0),
            log_pi=jnp.full(M, -jnp.log(M)),
        )
        # update_prior=False: the joint/conjugate fits are equal per SER, but the EB prior
        # M-step is a feedback loop that would amplify their ~1e-12 difference over sweeps.
        return finemap(model, L=3, ser_fit=ser, update_prior=False)

    c, j = fit("conjugate"), fit("joint")
    # The joint MAP fit is exact for a Gaussian likelihood -> identical PIPs and alphas.
    np.testing.assert_allclose(j.pip, c.pip, atol=1e-7)
    np.testing.assert_allclose(j.alpha, c.alpha, atol=1e-7)


def test_joint_recovers_causal_negbinom_high_L():
    X = _genotypes(jax.random.PRNGKey(2))
    N, M = X.shape
    causal, r = 11, 3.0
    mu = np.exp(1.5 + np.asarray(X[:, causal]) * 1.3)
    y = np.random.default_rng(2).negative_binomial(r, r / (r + mu)).astype(float)
    lik = likelihoods.neg_binomial_log(jnp.asarray(y).reshape(-1, 1), r=r)

    for ser in ("conjugate", "joint"):
        model = Model(
            design=SharedDesign(X=X),
            likelihood=lik,
            prior=priors.susie(1.0),
            log_pi=jnp.full(M, -jnp.log(M)),
            covariates=shared_covariates(jnp.ones((N, 1)), K=1, add_intercept=False),
        )
        res = finemap(model, L=10, coverage=0.95, purity=0.5, ser_fit=ser)
        assert not np.isnan(res.pip).any(), ser
        assert res.pip[causal] > 0.5, (ser, res.pip[causal])


def test_joint_equals_conjugate_for_mvsusie_mixture():
    """Joint must reproduce conjugate for a Gaussian-family, K>1, mixture-prior model."""
    key = jax.random.PRNGKey(3)
    X = _genotypes(key, N=200, M=25)
    N, M = X.shape
    K, causal = 2, 6
    B = np.zeros((M, K))
    B[causal] = [1.4, 1.0]
    Y = np.asarray(X @ B) + 0.5 * np.random.default_rng(3).normal(size=(N, K))
    comps = jnp.stack(
        [jnp.eye(K) * 0.1, jnp.eye(K), jnp.array([[1.0, 0.8], [0.8, 1.0]])]
    )

    def fit(ser):
        model = Model(
            design=SharedDesign(X=X),
            likelihood=likelihoods.mvn_resid(Y),
            prior=priors.mvsusie(comps),
            log_pi=jnp.full(M, -jnp.log(M)),
        )
        return finemap(model, L=3, ser_fit=ser, update_prior=False)

    c, j = fit("conjugate"), fit("joint")
    np.testing.assert_allclose(j.pip, c.pip, atol=1e-7)


def test_combine_joint_ridge_is_float32_safe_for_singular_components():
    """The joint fit must invert singular canonical components even in float32.

    ``combine_joint`` needs ``Sigma_g^{-1}``. The fully-shared canonical component
    ``[[1,1],[1,1]]`` is rank-1 singular, so it is regularised by ``Sigma_g + ridge I``.
    A *fixed* ``ridge = 1e-8`` silently fails in float32 -- ``1.0f + 1e-8 == 1.0f`` (float32
    ``eps`` is ``~1.2e-7``), so the component stays exactly singular and ``inv``/``slogdet``
    return ``inf``/``-inf``, poisoning the whole SER with ``nan`` (the notebook symptom).
    The dtype-relative default ridge (``sqrt(eps) * scale``) must keep every component
    finite. This runs on explicitly-float32 arrays so it exercises the bug regardless of
    the suite-wide float64 config.
    """
    from lasusie.model import Prior
    from lasusie.priors import canonical_components

    # The suite runs in float64 (conftest), where the old fixed 1e-8 ridge already worked;
    # toggle it off so this exercises the genuine float32 arithmetic (1f + 1e-8 == 1f).
    jax.config.update("jax_enable_x64", False)
    try:
        comps = canonical_components(K=2)  # float32; includes the rank-1 [[1,1],[1,1]]
        assert comps.dtype == jnp.float32
        prior = Prior(log_weights=jnp.zeros(comps.shape[0]), covariances=comps)
        # A well-conditioned concave like_term (a Gaussian potential in beta), all float32.
        b = jnp.array([1.0, 0.8])
        P = jnp.array([[2.0, 0.3], [0.3, 2.0]])
        like_term = lambda beta: -0.5 * (beta - b) @ (P @ (beta - b))  # noqa: E731

        post, log_marginal, warm = prior.combine_joint(like_term, jnp.zeros(2))
        assert np.isfinite(np.asarray(log_marginal)).all()
        assert np.isfinite(np.asarray(post.mean)).all()
        assert np.isfinite(np.asarray(post.covar)).all()
        assert np.isfinite(np.asarray(warm)).all()
    finally:
        jax.config.update("jax_enable_x64", True)
