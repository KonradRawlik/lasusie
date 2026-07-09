"""Fixed-effect covariate handling as a latent-predictor offset.

Nuisance covariates ``Z`` (intercept, age, sex, genotype PCs, batch, ...) enter the model
as an additive offset to the latent predictor::

    eta = design(B) + Z . gamma

The covariate coefficients ``gamma`` are a **point estimate** (a maximum-(expected-)
likelihood fixed effect): they carry no prior and no variance is propagated for them, the
standard treatment for nuisance covariates. They are re-estimated by a Newton M-step
between IBSS sweeps (:meth:`Covariates.update`), holding the current effect belief fixed --
a coordinate-ascent step on the same expected-log-likelihood term the ELBO monitors, so it
never decreases the ELBO. Unlike ``sushie``'s OLS regress-out (which is exact only for a
Gaussian likelihood), this latent-offset formulation is likelihood-agnostic: it works for
every likelihood and reduces exactly to regress-out in the Gaussian limit.

The offset geometry mirrors :mod:`lasusie.design` exactly:
  * :class:`SharedCovariates` -- multi-phenotype: one covariate matrix ``Z`` (``N x C``),
    per-phenotype coefficients ``gamma`` (``C x K``); ``offset = Z gamma`` is ``(N, K)``.
  * :class:`BlockCovariates`  -- multi-ancestry: stacked per-context ``Z`` (``K x N x C``),
    per-context coefficients ``gamma`` (``K x C``); ``offset[k] = Z[k] gamma[k]`` is ``(K, N)``.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .util import damped_newton as _newton


class Covariates(eqx.Module):
    """Abstract latent-offset covariate term ``Z . gamma``.

    Concrete subclasses store the covariate matrix ``Z`` (a constant) and the current
    coefficients ``gamma`` (updated by EB M-steps), and supply :meth:`offset` (the latent
    contribution) and :meth:`_with_gamma` (rebuild with new coefficients). :meth:`update`
    is shared: it Newton-maximises the (expected) log-likelihood over ``gamma``.
    """

    def offset(self) -> Array:
        """Latent contribution ``Z . gamma`` (design's latent shape)."""
        raise NotImplementedError

    def _offset_from(self, gamma: Array) -> Array:
        """Latent contribution for an arbitrary ``gamma`` (same shape as :attr:`gamma`)."""
        raise NotImplementedError

    def _with_gamma(self, gamma: Array) -> "Covariates":
        raise NotImplementedError

    def update(
        self,
        likelihood,
        eta_eff_mean: Array,
        eta_var: Array,
        propagate_variance: bool = True,
    ) -> "Covariates":
        """EB M-step: re-estimate ``gamma`` against the current effect belief.

        Maximises ``F(gamma) = E_q[log p(Y | eta_eff + Z gamma)]`` over ``gamma`` with the
        effect part ``eta_eff`` held fixed. When ``propagate_variance`` is ``True`` the
        objective is ``likelihood.expected_log_density`` (offset-uncertainty aware); when
        ``False`` it is the plain ``likelihood.log_density`` at the mean -- matching the SER's
        objective so covariate and effect updates optimise the same ELBO term.

        For a composite likelihood using the diagonal delta-method correction, the Hessian of
        that correction is ``stop_gradient``-ed on the mean (see
        :meth:`lasusie.model.AbstractLikelihood.expected_log_density`), so the ``gamma``-gradient
        sees only the plug-in log-density -- the same "corrected Laplace" decoupling used for
        the per-variant mode.
        """
        gamma0 = self._gamma()
        shape = gamma0.shape

        def f(flat_gamma):
            gamma = flat_gamma.reshape(shape)
            eta = eta_eff_mean + self._offset_from(gamma)
            if propagate_variance:
                return likelihood.expected_log_density(eta, eta_var)
            return likelihood.log_density(eta)

        gamma = _newton(f, gamma0.reshape(-1)).reshape(shape)
        return self._with_gamma(gamma)

    def _gamma(self) -> Array:
        raise NotImplementedError


class SharedCovariates(Covariates):
    """Multi-phenotype covariates: shared ``Z`` (``N x C``), coefficients ``gamma`` (``C x K``).

    ``offset = Z @ gamma`` has shape ``(N, K)`` -- matching :class:`lasusie.design.SharedDesign`.
    """

    Z: Array  # (N, C)
    gamma: Array  # (C, K)

    def offset(self) -> Array:
        return self.Z @ self.gamma  # (N, K)

    def _offset_from(self, gamma: Array) -> Array:
        return self.Z @ gamma

    def _with_gamma(self, gamma: Array) -> "SharedCovariates":
        return SharedCovariates(Z=self.Z, gamma=gamma)

    def _gamma(self) -> Array:
        return self.gamma


class BlockCovariates(Covariates):
    """Multi-ancestry covariates: stacked ``Z`` (``K x N x C``), coefficients ``gamma`` (``K x C``).

    ``offset[k] = Z[k] @ gamma[k]`` has shape ``(K, N)`` -- matching
    :class:`lasusie.design.BlockDesign`.
    """

    Z: Array  # (K, N, C)
    gamma: Array  # (K, C)

    def offset(self) -> Array:
        return jnp.einsum("knc,kc->kn", self.Z, self.gamma)  # (K, N)

    def _offset_from(self, gamma: Array) -> Array:
        return jnp.einsum("knc,kc->kn", self.Z, gamma)

    def _with_gamma(self, gamma: Array) -> "BlockCovariates":
        return BlockCovariates(Z=self.Z, gamma=gamma)

    def _gamma(self) -> Array:
        return self.gamma


def _prepend_intercept(Z: Array, axis: int) -> Array:
    """Prepend a column of ones along the covariate (last) axis."""
    ones_shape = list(Z.shape)
    ones_shape[-1] = 1
    return jnp.concatenate([jnp.ones(ones_shape, dtype=Z.dtype), Z], axis=-1)


def shared_covariates(Z: Array, K: int, add_intercept: bool = True) -> SharedCovariates:
    """Build :class:`SharedCovariates` for a multi-phenotype (``SharedDesign``) model.

    Args:
        Z: covariate matrix (shape ``(N, C)``).
        K: number of contexts (phenotypes); ``gamma`` is ``(C, K)``.
        add_intercept: prepend a ones column to ``Z`` (default ``True``).
    """
    Z = jnp.asarray(Z)
    if add_intercept:
        Z = _prepend_intercept(Z, axis=-1)
    C = Z.shape[-1]
    return SharedCovariates(Z=Z, gamma=jnp.zeros((C, K), dtype=Z.dtype))


def block_covariates(Z: Array, add_intercept: bool = True) -> BlockCovariates:
    """Build :class:`BlockCovariates` for a multi-ancestry (``BlockDesign``) model.

    Args:
        Z: stacked per-context covariate matrices (shape ``(K, N, C)``); ``K`` is inferred
            from the leading axis and ``gamma`` is ``(K, C)``.
        add_intercept: prepend a ones column to each context's ``Z`` (default ``True``).
    """
    Z = jnp.asarray(Z)
    if add_intercept:
        Z = _prepend_intercept(Z, axis=-1)
    K, _, C = Z.shape
    return BlockCovariates(Z=Z, gamma=jnp.zeros((K, C), dtype=Z.dtype))
