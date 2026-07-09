"""Constructors for the effect-size prior :class:`lasusie.model.Prior`.

Each SuSiE-family method is a particular mixture over covariance "sharing patterns" of
the K-vector effect. These helpers build the corresponding :class:`Prior`.
"""

import jax.numpy as jnp
from jax import Array

from .model import Prior


def susie(sigma0_sq: float) -> Prior:
    """Univariate SuSiE prior: a single Gaussian ``N(0, sigma0^2)`` (K=1)."""
    return Prior(
        log_weights=jnp.zeros(1),
        covariances=jnp.array([[[sigma0_sq]]]),
    )


def sushie(covariance: Array) -> Prior:
    """Multi-ancestry (sushie) prior: a single dense ``K x K`` component ``N(0, Sigma)``."""
    covariance = jnp.asarray(covariance)
    return Prior(
        log_weights=jnp.zeros(1),
        covariances=covariance[None, :, :],
    )


def mvsusie(components: Array, weights: Array | None = None) -> Prior:
    """Multi-phenotype (mvSuSiE) prior: a mixture of covariance components.

    Args:
        components: stack of ``G`` covariance matrices (shape ``(G, K, K)``) encoding
            sharing patterns (e.g. shared/rank-1, independent/diagonal, context-specific,
            null). Use :func:`canonical_components` to build the standard set.
        weights: optional initial mixing weights (shape ``(G,)``); defaults to uniform.
    """
    components = jnp.asarray(components)
    G = components.shape[0]
    log_weights = jnp.zeros(G) if weights is None else jnp.log(jnp.asarray(weights))
    return Prior(log_weights=log_weights, covariances=components)


def canonical_components(K: int, scale: float = 1.0) -> Array:
    """Standard mvSuSiE sharing patterns for ``K`` contexts (shape ``(G, K, K)``).

    Includes: fully shared (rank-1, perfectly correlated), independent (diagonal /
    identity), each context-specific pattern (active in one context only), and a null
    component (``Sigma -> 0``, represented with a tiny ridge for stability).
    """
    ones = jnp.ones((K, K))
    identity = jnp.eye(K)
    context_specific = jnp.stack([jnp.outer(e, e) for e in jnp.eye(K)])  # (K, K, K)
    null = 1e-8 * jnp.eye(K)
    shared = ones + 1e-8 * jnp.eye(K)  # ridge so it is strictly PD
    comps = jnp.concatenate(
        [shared[None], identity[None], context_specific, null[None]], axis=0
    )
    return scale * comps
