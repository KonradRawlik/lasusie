"""The design map: how a per-variant effect is applied to the latent predictor.

  * ``SharedDesign`` (one X shared across phenotypes) -- ``apply`` is ``X @ B`` and
    ``apply_effect(m, beta)`` is the rank-1 outer product ``outer(X[:, m], beta)``.
  * ``BlockDesign`` (one X per context) -- ``apply`` is the per-context ``X[k] @ B[:, k]``
    and ``apply_effect`` scales column m of each block by that context's coefficient.
"""

import jax
import jax.numpy as jnp
import numpy as np

from lasusie.design import BlockDesign, SharedDesign


def test_shared_design_apply_and_effect():
    N, M, K = 6, 4, 2
    X = jax.random.normal(jax.random.PRNGKey(3), (N, M))
    B = jax.random.normal(jax.random.PRNGKey(4), (M, K))
    d = SharedDesign(X=X)

    np.testing.assert_allclose(d.apply(B), X @ B, atol=1e-10)
    m, beta = 2, jnp.array([1.5, -0.5])
    np.testing.assert_allclose(d.apply_effect(m, beta), jnp.outer(X[:, m], beta), atol=1e-10)


def test_block_design_apply_and_effect():
    K, N, M = 3, 5, 4
    X = jax.random.normal(jax.random.PRNGKey(5), (K, N, M))
    B = jax.random.normal(jax.random.PRNGKey(6), (M, K))
    d = BlockDesign(X=X)

    expected = jnp.stack([X[k] @ B[:, k] for k in range(K)])  # (K, N)
    np.testing.assert_allclose(d.apply(B), expected, atol=1e-10)
    m, beta = 1, jnp.array([0.3, -1.0, 2.0])
    np.testing.assert_allclose(d.apply_effect(m, beta), X[:, :, m] * beta[:, None], atol=1e-10)
