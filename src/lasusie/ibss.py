"""Iterative Bayesian Stepwise Selection (IBSS) -- the SuSiE coordinate-ascent loop.

Each outer iteration sweeps over the ``L`` single effects. For effect ``l`` we remove its
current contribution from the latent belief ``q(eta) = N(mean, var)``, run a single-effect
regression against that offset, and add the updated effect back -- propagating both the
mean *and* the variance of the latent predictor (so the SER sees the uncertainty from the
other effects, recovering SuSiE's exact behaviour on the Gaussian special case).

Between sweeps, empirical-Bayes M-steps optionally update each effect's prior and the
likelihood's shared parameters. Convergence is monitored on the (approximate) ELBO
``E_q[log p(Y|eta)] - sum_l KL(q_l || prior_l)``.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import DTypeLike

from . import laplace
from .model import Model, kl_categorical
from .util import tree_stack


def _resolve_compute_dtype(model: Model, dtype: DTypeLike | None) -> np.dtype:
    """Pick the single floating dtype every array in a run should use.

    With ``dtype=None`` the dtype is the widest of the model's own floating arrays
    (``jnp.result_type`` over them): a uniformly-``float32`` model runs in ``float32`` rather
    than being silently promoted, while a model mixing precisions is levelled up to the widest
    so inference never *loses* precision. To pin a precision regardless of how the model was
    built (e.g. force ``float32`` even though ``log_pi`` came out ``float64``), pass ``dtype``
    explicitly.

    The chosen dtype is then canonicalised to what JAX will *actually* materialise: 64-bit
    floats collapse to 32-bit unless ``jax_enable_x64`` is set. Rather than let that happen
    silently (the trap this whole dtype pass exists to remove), a 64-bit dtype that would be
    downcast raises -- whether it was requested explicitly *or* carried by the model's own
    arrays -- so the caller enables x64 first instead of quietly losing precision.
    """
    explicit = dtype is not None
    if not explicit:
        leaves = [x for x in jax.tree_util.tree_leaves(model) if eqx.is_inexact_array(x)]
        dtype = jnp.result_type(*leaves) if leaves else jnp.float32
    requested = np.dtype(dtype)
    resolved = np.dtype(jax.dtypes.canonicalize_dtype(dtype))  # what JAX will really produce
    if requested.itemsize > resolved.itemsize:
        source = (
            f"dtype={requested.name} was requested"
            if explicit
            else f"the model holds {requested.name} arrays"
        )
        raise ValueError(
            f"{source} but JAX canonicalises {requested.name} to {resolved.name} because "
            f"64-bit values are disabled. Enable them with "
            f"jax.config.update('jax_enable_x64', True) before calling."
        )
    return resolved


def _cast_model(model: Model, dtype: np.dtype) -> Model:
    """Cast every floating array in the model pytree to ``dtype`` (ints/bools untouched)."""
    arrays, static = eqx.partition(model, eqx.is_inexact_array)
    arrays = jax.tree_util.tree_map(lambda x: x.astype(dtype), arrays)
    return eqx.combine(arrays, static)


class SuSiEResult(NamedTuple):
    """Output of :func:`ibss`.

    Attributes:
        alpha: inclusion probabilities per effect (shape ``(L, M)``).
        mean: per-variant posterior mean effect (shape ``(L, M, K)``).
        second: per-variant posterior second moment (shape ``(L, M, K, K)``).
        pip: posterior inclusion probability per variant (shape ``(M,)``).
        priors: the (possibly EB-updated) per-effect priors (a list of length ``L``).
        likelihood: the (possibly EB-updated) likelihood.
        covariates: the (possibly EB-updated) covariate term, or ``None`` if the model had
            no covariates. Its ``gamma`` holds the fitted covariate coefficients.
        elbo: final ELBO.
        elbo_history: ELBO after each sweep.
        iterations: number of outer sweeps run.
        converged: whether the ELBO change fell below ``tol``.
    """

    alpha: Array
    mean: Array
    second: Array
    pip: Array
    priors: list
    likelihood: object
    covariates: object
    elbo: float
    elbo_history: list
    iterations: int
    converged: bool


def _effect_latent(design, alpha_l, mean_l, second_l):
    """Latent mean and (per-coordinate) variance contributed by one single effect."""
    mean_effect = alpha_l[:, None] * mean_l  # (M, K)
    diag_second = jnp.diagonal(second_l, axis1=-2, axis2=-1)  # (M, K)
    diag_sq_effect = alpha_l[:, None] * diag_second  # (M, K)

    eta_mean = design.apply(mean_effect)
    eta_sq = design.apply_sq(diag_sq_effect)
    eta_var = eta_sq - eta_mean**2
    return eta_mean, eta_var


def _single_effect_regression(
    design, likelihood, prior, log_pi, offset_mean, offset_var, warm_modes,
    propagate_variance: bool = True,
    ser_fit: str = "joint",
):
    """One single-effect regression against a fixed offset belief ``q(eta)``.

    Args:
        propagate_variance: if ``True`` (default), use ``likelihood.expected_log_density``
            (the offset-uncertainty-aware objective -- Gauss-Hermite quadrature for
            pointwise likelihoods, the diagonal delta-method correction for composite
            ones). If ``False``, use the plain ``likelihood.log_density`` at the offset
            mean -- the zeroth-order approximation that ignores ``offset_var`` entirely.
            This choice is likelihood-agnostic: every likelihood implements both methods,
            so the switch lives here rather than per likelihood. Zeroth order is cheaper
            (skips quadrature / the Hessian the delta method needs) at the cost of ignoring
            uncertainty from the other effects.

    Returns ``(alpha, means, seconds, modes, kl_beta)`` where ``kl_beta`` is the per-variant
    ``KL(q(beta|m) || prior)`` used to assemble the effect's KL for the ELBO.
    """
    offset_var = jnp.maximum(offset_var, 0.0)  # guard tiny negatives from subtraction
    M = log_pi.shape[0]

    def fit_variant(m, x0):
        def g(beta):
            eta_mean = offset_mean + design.apply_effect(m, beta)
            if propagate_variance:
                return likelihood.expected_log_density(eta_mean, offset_var)
            return likelihood.log_density(eta_mean)

        if ser_fit == "conjugate":
            # One likelihood Laplace fit, prior folded in analytically (cheaper for a
            # mixture prior; leans on laplace.fit's guards for divergent count MLEs).
            pot = laplace.fit(g, x0)
            post, log_marginal = prior.combine(pot)
            return pot.mode, post, log_marginal
        # Default: regularise the fit with the prior at the source (MAP per component).
        post, log_marginal, warm = prior.combine_joint(g, x0)
        return warm, post, log_marginal

    modes, post, log_marginal = jax.vmap(fit_variant)(jnp.arange(M), warm_modes)
    alpha = jax.nn.softmax(log_pi + log_marginal)
    kl_beta = jax.vmap(prior.kl)(post)  # (M,)
    return alpha, post, modes, kl_beta


# Public, jitted entry point (the effect-loop scan calls the un-jitted core above directly,
# so nesting this jit inside the scan would be redundant). Kept as a standalone jitted
# function for callers that run a single SER in isolation.
single_effect_regression = eqx.filter_jit(_single_effect_regression)


@eqx.filter_jit
def _run_sweep(
    design, likelihood, log_pi, priors_stacked,
    eta_mean, eta_var, alpha, mean, second, modes,
    propagate_variance, ser_fit, update_prior,
):
    """One IBSS sweep over all ``L`` effects, lowered to a single ``jax.lax.scan``.

    The effect loop's only cross-effect coupling is through the latent belief
    ``(eta_mean, eta_var)`` -- effect ``l`` reads only its own ``alpha[l]/mean[l]/second[l]``
    (remove step) and ``modes[l]`` (warm start). So it lowers faithfully to a scan carrying
    ``(eta_mean, eta_var)`` with the per-effect state as ``xs``. ``priors_stacked`` is a single
    :class:`~lasusie.model.Prior` pytree with a leading ``L`` axis (built once via
    :func:`~lasusie.util.tree_stack`); scan slices it to feed each effect its own prior and
    stacks the (possibly EB-updated) priors back. Compiled once and reused across sweeps.
    """
    def body(carry, xs):
        eta_mean, eta_var = carry
        alpha_l, mean_l, second_l, modes_l, prior_l = xs

        # remove effect l's current contribution from the latent belief
        em, ev = _effect_latent(design, alpha_l, mean_l, second_l)
        eta_mean = eta_mean - em
        eta_var = eta_var - ev

        a_l, post_l, md_l, kl_beta = _single_effect_regression(
            design, likelihood, prior_l, log_pi, eta_mean, eta_var, modes_l,
            propagate_variance, ser_fit,
        )
        mu_l = post_l.mixture_mean
        s_l = post_l.mixture_second_moment
        kl_l = kl_categorical(a_l, log_pi) + a_l @ kl_beta

        if update_prior:
            if prior_l.covariances.shape[0] == 1:
                # Single-component EB (susie/sushie): alpha-weighted 2nd moment.
                prior_l = _update_prior_from_moments(prior_l, a_l, mu_l, s_l)
            else:
                # Mixture EB (mvSuSiE): EM M-step over components and weights.
                prior_l = prior_l.update(a_l, post_l)

        # add the updated effect back
        em, ev = _effect_latent(design, a_l, mu_l, s_l)
        eta_mean = eta_mean + em
        eta_var = eta_var + ev

        return (eta_mean, eta_var), (a_l, mu_l, s_l, md_l, kl_l, prior_l)

    xs = (alpha, mean, second, modes, priors_stacked)
    (eta_mean, eta_var), ys = jax.lax.scan(body, (eta_mean, eta_var), xs)
    alpha, mean, second, modes, effect_kl, priors_stacked = ys
    return eta_mean, eta_var, alpha, mean, second, modes, effect_kl, priors_stacked


def ibss(
    model: Model,
    L: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    update_prior: bool = True,
    update_likelihood: bool = True,
    update_covariates: bool = True,
    propagate_variance: bool = True,
    ser_fit: str = "joint",
    dtype: DTypeLike | None = None,
) -> SuSiEResult:
    """Run IBSS for ``L`` single effects.

    Args:
        model: a fully specified :class:`~lasusie.model.Model`.
        L: number of single effects.
        max_iter: maximum outer sweeps.
        tol: convergence tolerance on the ELBO change between sweeps.
        dtype: floating precision for the whole run. ``None`` (default) infers it from the
            model's own arrays, so the dtype the model was built with is honoured. Pass an
            explicit dtype (e.g. ``jnp.float32``) to force it -- the model is cast and every
            internal array is created at that precision. Note that 64-bit precision also
            requires ``jax.config.update("jax_enable_x64", True)``; a 64-bit dtype that JAX
            would downcast -- whether passed here or carried by the model's arrays -- raises
            rather than silently losing precision.
        update_prior: whether to EB-update each effect's prior between sweeps.
        update_likelihood: whether to EB-update the likelihood parameters between sweeps.
        update_covariates: whether to re-estimate the covariate coefficients ``gamma``
            between sweeps (no effect if the model has no covariates).
        propagate_variance: whether the SER accounts for offset uncertainty (see
            :func:`single_effect_regression`). Also controls whether the reported ELBO's
            expected-log-likelihood term uses ``expected_log_density`` or the zeroth-order
            ``log_density``, so the objective being monitored matches what was optimized.
        ser_fit: how each single-effect regression combines the likelihood with the prior.
            ``"joint"`` (default) fits the MAP of the joint per prior component (see
            :meth:`~lasusie.model.Prior.combine_joint`) -- the prior curvature regularises a
            divergent single-effect MLE at the source, so it is robust for non-Gaussian
            likelihoods, at the cost of one Laplace fit per component (one total for the
            single-component ``susie``/``sushie`` priors). ``"conjugate"`` instead Laplace-
            approximates the likelihood once and folds in the prior in closed form; it is
            cheaper for a mixture prior but relies on numerical guards for count/exp-family
            likelihoods. The two are equal for a Gaussian likelihood.
    """
    # Resolve one floating precision for the whole run and cast the model to it, so every
    # array below -- model data and the internal state alike -- shares a single dtype and
    # nothing silently promotes (the failure mode this dtype pass removes).
    compute_dtype = _resolve_compute_dtype(model, dtype)
    model = _cast_model(model, compute_dtype)

    M = model.n_variants
    K = model.prior.covariances.shape[-1]
    design = model.design
    log_pi = model.log_pi

    likelihood = model.likelihood
    # All L effects share the same starting prior; stack them into one Prior pytree with a
    # leading L axis so the per-effect sweep is a single scan. The (possibly EB-updated)
    # stack is carried across sweeps and unstacked to a list only at the end.
    priors_stacked = tree_stack([model.prior] * L)
    covariates = model.covariates

    alpha = jnp.zeros((L, M), compute_dtype)
    mean = jnp.zeros((L, M, K), compute_dtype)
    second = jnp.zeros((L, M, K, K), compute_dtype)
    modes = jnp.zeros((L, M, K), compute_dtype)

    latent_shape = design.apply(jnp.zeros((M, K), compute_dtype)).shape
    # The latent belief carries the covariate offset in its mean throughout: the per-effect
    # remove/add-back (``_effect_latent``) only touches the effect part, so ``cov_offset``
    # persists in ``eta_mean`` and every SER is solved conditional on the covariates.
    cov_offset = (
        jnp.zeros(latent_shape, compute_dtype) if covariates is None else covariates.offset()
    )
    eta_mean = cov_offset
    eta_var = jnp.zeros(latent_shape, compute_dtype)

    elbo_history = []
    converged = False
    it = 0
    while it < max_iter:
        it += 1

        # One fused sweep over all L effects (see _run_sweep). Only the effect loop is
        # compiled; the covariate/likelihood/ELBO steps below stay in Python.
        (
            eta_mean, eta_var, alpha, mean, second, modes, effect_kl, priors_stacked
        ) = _run_sweep(
            design, likelihood, log_pi, priors_stacked,
            eta_mean, eta_var, alpha, mean, second, modes,
            propagate_variance, ser_fit, update_prior,
        )

        if covariates is not None and update_covariates:
            # M-step for the covariate coefficients against the fixed effect belief. The
            # effect part is ``eta_mean - cov_offset``; re-fit gamma, then reinstate the
            # (updated) covariate offset in the latent mean.
            eta_eff_mean = eta_mean - cov_offset
            covariates = covariates.update(
                likelihood, eta_eff_mean, eta_var, propagate_variance
            )
            cov_offset = covariates.offset()
            eta_mean = eta_eff_mean + cov_offset

        if update_likelihood:
            likelihood = likelihood.updated(eta_mean, eta_var)

        if propagate_variance:
            exp_ll = likelihood.expected_log_density(eta_mean, jnp.maximum(eta_var, 0.0))
        else:
            exp_ll = likelihood.log_density(eta_mean)
        elbo = float(exp_ll - jnp.sum(effect_kl))
        elbo_history.append(elbo)

        if it > 1 and abs(elbo_history[-1] - elbo_history[-2]) < tol:
            converged = True
            break

    pip = 1.0 - jnp.prod(1.0 - alpha, axis=0)
    # Unstack the carried Prior pytree back into the documented list of L per-effect priors.
    priors = [jax.tree_util.tree_map(lambda leaf: leaf[i], priors_stacked) for i in range(L)]
    return SuSiEResult(
        alpha=alpha,
        mean=mean,
        second=second,
        pip=pip,
        priors=priors,
        likelihood=likelihood,
        covariates=covariates,
        elbo=elbo_history[-1],
        elbo_history=elbo_history,
        iterations=it,
        converged=bool(converged),
    )


def _update_prior_from_moments(prior, alpha_l, mean_l, second_l):
    """EB update of a single-component prior from the effect's alpha-weighted 2nd moment.

    ``Sigma <- sum_m alpha_m E[beta beta^T | m] / sum_m alpha_m``. (For a single-component
    prior this matches sushie's ``weighted_sum_covar``; mixture EB is handled in Phase 5.)
    """
    Sigma = jnp.einsum("m,mkl->kl", alpha_l, second_l) / alpha_l.sum()
    return type(prior)(log_weights=prior.log_weights, covariances=Sigma[None])
