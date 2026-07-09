"""The likelihood class hierarchy: base defaults and subclass overrides.

``AbstractLikelihood`` supplies composite-case defaults (dense ``hessian_diagonal``,
delta-method ``expected_log_density``, identity ``updated``) on top of an abstract
``log_density``. Subclasses override only where their structure simplifies things:
  * ``PointwiseLikelihood`` -> exact Gauss-Hermite ``expected_log_density`` and an
    elementwise (O(N)) ``hessian_diagonal``;
  * ``CoxPH`` -> a structured O(N) ``hessian_diagonal``, inheriting the delta-method
    ``expected_log_density`` and identity ``updated`` unchanged.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from lasusie import AbstractLikelihood, likelihoods
from lasusie.model import PointwiseLikelihood


class _Coupled(AbstractLikelihood):
    """Minimal composite likelihood: ``log p = -0.5 eta^T A eta`` (A couples all eta)."""

    A: Array

    def log_density(self, eta: Array) -> Array:
        e = eta.reshape(-1)
        return -0.5 * e @ self.A @ e


def test_abstract_likelihood_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractLikelihood()


def test_minimal_subclass_inherits_working_defaults():
    # A subclass implementing ONLY log_density gets a correct delta-method
    # expected_log_density (via the dense hessian_diagonal fallback) and identity updated.
    key = jax.random.PRNGKey(0)
    N = 12
    B = jax.random.normal(key, (N, N))
    A = B @ B.T + jnp.eye(N)  # symmetric PD, dense (genuinely coupled)
    lik = _Coupled(A=A)

    eta = jax.random.normal(jax.random.PRNGKey(1), (N,))
    var = jax.nn.softplus(jax.random.normal(jax.random.PRNGKey(2), (N,)))

    # Hessian of -0.5 eta^T A eta is -A; its diagonal is -diag(A).
    np.testing.assert_allclose(
        np.asarray(lik.hessian_diagonal(eta)), -np.asarray(jnp.diag(A)), atol=1e-5
    )
    expected = float(lik.log_density(eta)) + 0.5 * float(jnp.sum(-jnp.diag(A) * var))
    np.testing.assert_allclose(float(lik.expected_log_density(eta, var)), expected, rtol=1e-6)
    # No shared parameters -> updated is the inherited identity.
    assert lik.updated(eta, var) is lik


def test_correction_is_stop_gradient_decoupled_from_mode():
    # The delta correction must not enter the gradient of the mode-finding objective:
    # grad_eta expected_log_density == grad_eta log_density (the "corrected Laplace"
    # decoupling), inherited by every AbstractLikelihood.
    N = 8
    A = jnp.eye(N) * 2.0
    lik = _Coupled(A=A)
    eta = jax.random.normal(jax.random.PRNGKey(3), (N,))
    var = jnp.full(N, 0.7)

    g_corr = jax.grad(lambda e: lik.expected_log_density(e, var))(eta)
    g_plain = jax.grad(lik.log_density)(eta)
    np.testing.assert_allclose(np.asarray(g_corr), np.asarray(g_plain), atol=1e-6)


@pytest.mark.parametrize(
    "lik_fn",
    [
        lambda: likelihoods.gaussian(
            jax.random.normal(jax.random.PRNGKey(4), (15, 1)), sigma2=0.8
        ),
        lambda: likelihoods.poisson_log(
            jax.random.poisson(jax.random.PRNGKey(5), 2.0, (15, 1)).astype(float)
        ),
    ],
)
def test_pointwise_hessian_diagonal_matches_dense(lik_fn):
    # The elementwise O(N) override must equal the dense diagonal it replaces.
    lik = lik_fn()
    eta = jax.random.normal(jax.random.PRNGKey(6), lik.y.shape) * 0.5
    dense = jnp.diagonal(jax.hessian(lambda e: lik.log_density(e.reshape(eta.shape)))(eta.reshape(-1)))
    np.testing.assert_allclose(
        np.asarray(lik.hessian_diagonal(eta)).reshape(-1), np.asarray(dense), atol=1e-5
    )


def test_cox_inherits_base_expected_log_density_and_updated():
    # CoxPH implements only log_density + hessian_diagonal; the other two come from the base.
    assert likelihoods.CoxPH.expected_log_density is AbstractLikelihood.expected_log_density
    assert likelihoods.CoxPH.updated is AbstractLikelihood.updated
    # And PointwiseLikelihood really does override both simplifiable methods.
    assert PointwiseLikelihood.expected_log_density is not AbstractLikelihood.expected_log_density
    assert PointwiseLikelihood.hessian_diagonal is not AbstractLikelihood.hessian_diagonal
