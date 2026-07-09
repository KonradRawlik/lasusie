"""The frozen-metric Laplace fit.

For a quadratic log-density ``f(b) = c - 0.5 (b - m)^T P (b - m)`` the Laplace
approximation is exact, so ``laplace.fit`` must return ``mode = m``, ``precision = P``,
``log_scale = c`` regardless of the warm-start -- and it must vmap cleanly over many
variants (this is the closed-form == Laplace check underpinning the Gaussian path).
For a non-quadratic objective whose MLE runs off to infinity, the fit must instead stay
finite: it bounds the mode and floors the precision rather than returning inf/singular.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lasusie import laplace


def quadratic(mode, precision, const):
    def f(b):
        d = b - mode
        return const - 0.5 * d @ precision @ d

    return f


@pytest.mark.parametrize("K", [1, 2, 4])
def test_fit_recovers_gaussian(K):
    key = jax.random.PRNGKey(K)
    k_m, k_a, k_x = jax.random.split(key, 3)
    mode = jax.random.normal(k_m, (K,))
    A = jax.random.normal(k_a, (K, K))
    precision = A @ A.T + K * jnp.eye(K)  # symmetric positive definite
    const = 1.234

    x0 = jax.random.normal(k_x, (K,))  # deliberately far-ish warm start
    pot = laplace.fit(quadratic(mode, precision, const), x0)

    np.testing.assert_allclose(pot.mode, mode, atol=1e-8)
    np.testing.assert_allclose(pot.precision, precision, atol=1e-8)
    np.testing.assert_allclose(pot.log_scale, const, atol=1e-8)


def test_fit_vmaps_over_variants():
    K, M = 2, 16
    key = jax.random.PRNGKey(0)
    k_m, k_a = jax.random.split(key)
    modes = jax.random.normal(k_m, (M, K))
    As = jax.random.normal(k_a, (M, K, K))
    precisions = jnp.einsum("mij,mkj->mik", As, As) + K * jnp.eye(K)

    def fit_one(mode, precision):
        f = lambda b: -0.5 * (b - mode) @ precision @ (b - mode)  # noqa: E731
        return laplace.fit(f, jnp.zeros(K))

    pots = jax.vmap(fit_one)(modes, precisions)
    np.testing.assert_allclose(pots.mode, modes, atol=1e-7)
    np.testing.assert_allclose(pots.precision, precisions, atol=1e-7)


def test_laplace_bounds_divergent_mode_and_floors_precision():
    """A monotone objective has its MLE at infinity; the fit must stay finite."""
    f = lambda b: jnp.sum(3.0 * b)  # noqa: E731  linear -> gradient never vanishes
    pot = laplace.fit(f, jnp.zeros(1), mode_bound=1e2, precision_floor=1e-10)
    assert np.isfinite(np.asarray(pot.mode)).all()
    assert abs(float(pot.mode[0])) <= 1e2 + 1e-6  # clipped to the bound
    assert np.isfinite(float(pot.log_scale))
    # Curvature of a linear function is 0; the floor keeps the precision strictly positive.
    assert float(pot.precision[0, 0]) >= 1e-10
    assert np.isfinite(float(pot.precision[0, 0]))
