"""The design map: how the ``M x K`` effect array ``B`` becomes the latent predictor.

Multi-ancestry and multi-phenotype are two configs of one linear operator ``eta = X(B)``.
A :class:`DesignMap` supplies just the forward map :meth:`apply` and the per-variant
single-effect contribution :meth:`apply_effect`; gradients/precisions in the SER come
from autodiff through these.

  * :class:`SharedDesign` -- multi-phenotype: one genotype ``X`` shared across contexts,
    every individual measured for all K phenotypes.  ``eta = X B`` has shape ``(N, K)``.
  * :class:`BlockDesign`  -- multi-ancestry: disjoint individuals per context, stacked
    genotypes ``X`` of shape ``(K, N, M)``.  ``eta[k] = X[k] B[:, k]`` has shape ``(K, N)``.

K=1 is :class:`SharedDesign` with a single column. (Unequal per-context sample sizes in
:class:`BlockDesign` are handled by padding + masking; not yet implemented.)
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class DesignMap(eqx.Module):
    """Abstract linear map from the effect array ``B`` (``M x K``) to the latent ``eta``."""

    def apply(self, B: Array) -> Array:
        raise NotImplementedError

    def apply_effect(self, m: Array, beta: Array) -> Array:
        """Latent contribution of a single effect at variant ``m`` with effect ``beta``.

        Equivalent to ``apply(e_m beta^T)`` but formed directly. ``m`` is a variant index,
        ``beta`` has shape ``(K,)``; the return has the design's latent shape.
        """
        raise NotImplementedError

    def apply_sq(self, A: Array) -> Array:
        """``apply`` with the genotype entries squared -- used for the latent second moment.

        ``E[eta_l^2]`` per coordinate ``= apply_sq(alpha_m * diag E[beta beta^T]_m)``.
        """
        raise NotImplementedError

    def min_abs_corr(self, idx: Array) -> Array:
        """Purity of a variant set: the minimum absolute pairwise genotype correlation.

        Returns 1.0 for a singleton set. For multi-context designs the per-context purity
        is averaged (equal weight, i.e. equal sample sizes).
        """
        raise NotImplementedError


def _min_abs_corr(X_cols: Array) -> Array:
    """Minimum absolute off-diagonal correlation among columns of ``X_cols`` (``N x S``)."""
    if X_cols.shape[1] <= 1:
        return jnp.asarray(1.0)
    R = jnp.corrcoef(X_cols, rowvar=False)
    S = R.shape[0]
    off = jnp.abs(R)[~jnp.eye(S, dtype=bool)]
    return jnp.min(off)


class SharedDesign(DesignMap):
    """Multi-phenotype design: shared genotype ``X`` (``N x M``); ``eta = X B`` (``N x K``)."""

    X: Array

    def apply(self, B: Array) -> Array:
        return self.X @ B

    def apply_effect(self, m: Array, beta: Array) -> Array:
        return jnp.outer(self.X[:, m], beta)  # (N, K)

    def apply_sq(self, A: Array) -> Array:
        return (self.X**2) @ A

    def min_abs_corr(self, idx: Array) -> Array:
        return _min_abs_corr(self.X[:, idx])


class BlockDesign(DesignMap):
    """Multi-ancestry design: stacked disjoint genotypes ``X`` (``K x N x M``).

    ``eta[k] = X[k] @ B[:, k]``, shape ``(K, N)``.
    """

    X: Array  # (K, N, M)

    def apply(self, B: Array) -> Array:
        # eta[k, n] = sum_m X[k, n, m] B[m, k]
        return jnp.einsum("knm,mk->kn", self.X, B)

    def apply_effect(self, m: Array, beta: Array) -> Array:
        return self.X[:, :, m] * beta[:, None]  # (K, N)

    def apply_sq(self, A: Array) -> Array:
        return jnp.einsum("knm,mk->kn", self.X**2, A)

    def min_abs_corr(self, idx: Array) -> Array:
        # per-context purity, averaged across contexts (equal sample sizes assumed)
        per_context = jnp.stack([_min_abs_corr(self.X[k][:, idx]) for k in range(self.X.shape[0])])
        return jnp.mean(per_context)
