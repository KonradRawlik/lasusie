"""Core model components for lasusie.

This module grows across the implementation phases. It currently holds the flexible
**prior**: a mixture of zero-mean Gaussians over the K-vector effect ``beta``, which is
conjugate to the Gaussian likelihood potential produced by the nested Laplace step, so
the single-effect "combine potential x prior -> posterior + marginal likelihood" step is
closed form.

The mixture subsumes the whole SuSiE family as one code path:
  * univariate SuSiE .... K=1, one component ``N(0, sigma0^2)``
  * sushie (ancestry) ... K>1, one dense component ``N(0, Sigma)``
  * mvSuSiE (phenotype) . K>1, a mixture of covariance "sharing patterns"
  * point-normal ........ a component with ``Sigma -> 0``
"""

import abc
import functools
from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp

from .covariates import Covariates
from .design import DesignMap


class SERPosterior(NamedTuple):
    """Per-variant posterior over ``beta`` from combining a potential with the prior.

    A mixture of ``G`` Gaussians (one per prior component), with mixing weights given by
    the component responsibilities. Arrays are unbatched (a single variant); the SER
    ``vmap``s :meth:`Prior.combine` over variants, adding a leading ``M`` axis.

    Attributes:
        resp: component responsibilities ``r_g = w_g Z_g / sum_g w_g Z_g`` (shape ``(G,)``).
        mean: per-component posterior mean (shape ``(G, K)``).
        covar: per-component posterior covariance (shape ``(G, K, K)``).
    """

    resp: Array
    mean: Array
    covar: Array

    @property
    def mixture_mean(self) -> Array:
        """Posterior mean ``E[beta | variant]`` (shape ``(K,)``)."""
        return jnp.einsum("...g,...gk->...k", self.resp, self.mean)

    @property
    def mixture_second_moment(self) -> Array:
        """Posterior second moment ``E[beta beta^T | variant]`` (shape ``(K, K)``)."""
        second = self.covar + jnp.einsum("...gk,...gl->...gkl", self.mean, self.mean)
        return jnp.einsum("...g,...gkl->...kl", self.resp, second)


def _kl_mvn(m0: Array, S0: Array, S1: Array) -> Array:
    """``KL( N(m0, S0) || N(0, S1) )`` for the posterior-vs-prior effect term."""
    K = S1.shape[-1]
    S1_inv = jnp.linalg.inv(S1)
    _, logdet_S1 = jnp.linalg.slogdet(S1)
    _, logdet_S0 = jnp.linalg.slogdet(S0)
    trace = jnp.trace(S1_inv @ S0)
    quad = m0 @ S1_inv @ m0
    return 0.5 * (trace + quad - K + logdet_S1 - logdet_S0)


def kl_categorical(alpha: Array, log_pi: Array) -> Array:
    """``KL( Cat(alpha) || Cat(pi) )`` (nan-safe at ``alpha_m = 0``)."""
    return jnp.sum(jnp.where(alpha > 0, alpha * (jnp.log(alpha) - log_pi), 0.0))


class AbstractLikelihood(eqx.Module):
    """Base class for observation likelihoods ``log p(Y | eta)``.

    A likelihood binds its data as pytree fields and exposes three methods to the SER /
    IBSS loop: :meth:`log_density`, :meth:`expected_log_density`, and :meth:`updated`.
    Only :meth:`log_density` is abstract; the rest have defaults here that are correct for
    an arbitrarily *coupled* (composite) likelihood, so a new likelihood in the general
    case needs to implement nothing else. Subclasses override a default only where their
    structure admits something better:

      * a *factorised* likelihood (:class:`PointwiseLikelihood`) replaces
        :meth:`expected_log_density` with exact per-observation Gauss-Hermite quadrature
        and :meth:`hessian_diagonal` with the elementwise second derivative;
      * a likelihood with a closed-form offset expectation (``MVNResidual``) overrides
        :meth:`expected_log_density` directly;
      * a likelihood with structured curvature (``CoxPH``) overrides only
        :meth:`hessian_diagonal` and inherits the delta-method
        :meth:`expected_log_density` built on top of it.
    """

    @abc.abstractmethod
    def log_density(self, eta: Array) -> Array:
        """Full ``log p(Y | eta)`` (a scalar); may couple observations arbitrarily."""
        raise NotImplementedError

    def hessian_diagonal(self, eta: Array) -> Array:
        """Diagonal of ``Hessian(log_density)(eta)`` (same shape as ``eta``).

        Generic fallback: materialise the full ``N x N`` Hessian via :func:`jax.hessian`
        and read its diagonal. Correct for any ``log_density`` but ``O(N^2)`` in time and
        memory -- the bottleneck for large-``N`` composite likelihoods. A subclass whose
        curvature diagonal has a cheaper closed form (e.g. ``CoxPH``'s risk-set cumulative
        sums, or a factorised likelihood's elementwise second derivative) overrides this.
        """
        shape = eta.shape

        def flat_log_density(flat_eta):
            return self.log_density(flat_eta.reshape(shape))

        return jnp.diagonal(jax.hessian(flat_log_density)(eta.reshape(-1))).reshape(shape)

    def expected_log_density(self, mean: Array, var: Array) -> Array:
        """Delta-method ``E_{eta~q}[log_density]`` for ``q(eta) = N(mean, diag(var))``.

        Composite likelihoods can't use per-observation quadrature (the density doesn't
        split into per-observation terms), so this approximates the expectation to second
        order:

            E_q[log_density] ~= log_density(mean) + 0.5 * sum_n H_nn(mean) * var_n,

        using only the diagonal of ``H = Hessian(log_density)`` (the belief is
        diagonal-covariance) via :meth:`hessian_diagonal`.

        The Hessian diagonal is evaluated at a ``stop_gradient``-ed copy of ``mean``, so
        the correction contributes to the *value* (the SER's marginal likelihood / ELBO)
        but not to the gradient/Hessian JAX computes when this is the mode-finding
        objective in :func:`lasusie.laplace.fit`. So the mode is located from the plug-in
        (point-estimate) log-density alone and the correction only adjusts the evidence --
        the standard "corrected Laplace" decoupling. This is deliberate: differentiating
        the correction itself would need 3rd/4th-order derivatives of ``log_density`` for a
        term that is already ``O(var)``, a higher-order effect on the mode's location than
        the approximation tracks elsewhere.
        """
        base = self.log_density(mean)
        h_diag = self.hessian_diagonal(jax.lax.stop_gradient(mean))
        return base + 0.5 * jnp.sum(h_diag.reshape(-1) * var.reshape(-1))

    def updated(self, eta_mean: Array, eta_var: Array) -> "AbstractLikelihood":
        """Empirical-Bayes refresh of shared parameters; identity when there are none."""
        return self


class Prior(eqx.Module):
    """Mixture of zero-mean Gaussians over the effect ``beta in R^K``.

    ``pi(beta) = sum_g w_g N(beta; 0, Sigma_g)``, with ``w_g = softmax(log_weights)_g``.

    Attributes:
        log_weights: unnormalised log mixing weights (shape ``(G,)``).
        covariances: component covariances ``Sigma_g`` (shape ``(G, K, K)``).
    """

    log_weights: Array
    covariances: Array

    @property
    def log_mix(self) -> Array:
        """Normalised log mixing weights ``log w_g`` (shape ``(G,)``)."""
        return self.log_weights - logsumexp(self.log_weights)

    def combine(self, potential) -> tuple[SERPosterior, Array]:
        """Combine a single-variant Gaussian potential with the prior (closed form).

        The potential represents the (Laplace-approximated) likelihood as a function of
        ``beta``:  ``L(beta) prop exp(log_scale - 0.5 (beta - mode)^T P (beta - mode))``,
        i.e. a Gaussian *shape* with precision ``P = precision`` centred at ``mode``.

        For each prior component ``N(0, Sigma_g)`` we form the conjugate posterior and
        the component marginal ``Z_g = int L(beta) N(beta; 0, Sigma_g) dbeta``.  Writing
        ``A = P^{-1}`` (the potential covariance), ``C_g = A + Sigma_g``:

            post_mean_g  = Sigma_g C_g^{-1} mode
            post_covar_g = A C_g^{-1} Sigma_g
            log Z_g      = log_scale - 0.5 logdet(P) - 0.5 logdet(C_g)
                                     - 0.5 mode^T C_g^{-1} mode

        These forms never invert ``Sigma_g``, so a null/zeroed component (``Sigma_g=0``)
        is well behaved: it yields ``post=0`` and ``log Z = log_scale - 0.5 mode^T P mode``
        (the likelihood at ``beta = 0`` -- the correct null evidence).

        Returns:
            ``(posterior, log_marginal)`` where ``log_marginal = log sum_g w_g Z_g``.
        """
        P = potential.precision
        b = potential.mode
        A = jnp.linalg.inv(P)
        _, logdet_P = jnp.linalg.slogdet(P)

        def per_component(Sigma):
            C = A + Sigma
            Cinv = jnp.linalg.inv(C)
            _, logdet_C = jnp.linalg.slogdet(C)
            post_mean = Sigma @ (Cinv @ b)
            post_covar = A @ Cinv @ Sigma
            log_Z = (
                potential.log_scale
                - 0.5 * logdet_P
                - 0.5 * logdet_C
                - 0.5 * b @ (Cinv @ b)
            )
            return post_mean, post_covar, log_Z

        post_mean, post_covar, log_Z = eqx.filter_vmap(per_component)(self.covariances)

        log_num = self.log_mix + log_Z
        log_marginal = logsumexp(log_num)
        resp = jnp.exp(log_num - log_marginal)

        return SERPosterior(resp=resp, mean=post_mean, covar=post_covar), log_marginal

    def combine_joint(
        self, like_term: Callable[[Array], Array], x0: Array, ridge: float | None = None
    ) -> tuple[SERPosterior, Array, Array]:
        """Joint (MAP) alternative to :meth:`combine`.

        Instead of Laplace-approximating the likelihood *once* and folding in the prior in
        closed form, this Laplace-approximates the **joint** ``exp(like_term(beta)) *
        N(beta; 0, Sigma_g)`` separately for each component -- i.e. it fits the MAP of
        ``h_g(beta) = like_term(beta) - 0.5 beta^T Sigma_g^{-1} beta`` and expands there.

        This regularises a divergent single-effect MLE at the source: a count / exponential-
        family null variant makes ``like_term`` monotone (no interior maximum), but the prior
        curvature ``Sigma_g^{-1}`` makes every ``h_g`` strictly concave, so the mode is finite
        and the precision positive-definite without needing the numerical guards
        :func:`lasusie.laplace.fit` falls back on. It costs one Laplace fit *per component*
        (vs one total for :meth:`combine`), and equals :meth:`combine` exactly for a Gaussian
        likelihood (the joint is then exactly Gaussian).

        Because it needs ``Sigma_g^{-1}`` it adds a small ``ridge`` (``Sigma_g -> Sigma_g +
        ridge I``) so that *singular* prior components -- the rank-deficient canonical mvSuSiE
        covariances (e.g. a per-context "singleton" ``[[1,0],[0,0]]``, the fully-shared rank-1
        ``[[1,1],[1,1]]``) and zero components -- stay invertible. A singular component means
        ``beta`` is pinned to 0 in the null directions (infinite prior precision there); the
        ridge realises that as a large finite penalty, and it cancels in the marginal below
        (``-0.5 logdet(Sigma_ridge) - 0.5 logdet(precision) = -0.5 logdet(I + Sigma_ridge
        P_like)``), so the result is accurate to ``O(ridge)`` on the constrained subspace.

        The default ``ridge`` is **dtype- and scale-relative**: ``sqrt(eps) * max(|Sigma|, 1)``
        where ``eps`` is the machine epsilon of the component dtype. This matters because a
        fixed absolute ridge fails silently in float32: a rank-1 component like ``[[1,1],
        [1,1]]`` stays *exactly* singular after ``Sigma + 1e-8 I`` because ``1.0f + 1e-8 ==
        1.0f`` (float32 ``eps`` is ``~1.2e-7``), so ``inv``/``slogdet`` return ``inf``/``-inf``
        and poison the whole SER. Tying the ridge to ``sqrt(eps)`` (``~3.4e-4`` in float32,
        ``~1.5e-8`` in float64) keeps it above the representable floor in the working precision
        yet negligible in float64 -- the joint fit still reproduces :meth:`combine` for a
        Gaussian likelihood there. Pass an explicit ``ridge`` to override.

        The component marginal is the standard Laplace form (the ``(2 pi)^{K/2}`` factors of
        the prior normaliser and the Laplace integral cancel)::

            log Z_g = h_g(beta_hat_g) - 0.5 logdet(Sigma_g) - 0.5 logdet(precision_g)

        Returns ``(posterior, log_marginal, warm)`` -- ``warm`` is the responsibility-weighted
        posterior mean, a warm start for the next sweep's fits (mirroring the potential mode
        that :meth:`combine`'s caller carries).
        """
        from . import laplace  # local import: laplace has no model dependency (avoid cycle)

        K = self.covariances.shape[-1]
        dtype = self.covariances.dtype
        if ridge is None:
            # Scale it to the largest component magnitude so the ridge survives float32
            # (where 1 + 1e-8 == 1) on a unit-scale singular component, and floor the scale
            # at 1 so a mixture of only tiny/near-null components still gets a real ridge.
            scale = jnp.maximum(jnp.max(jnp.abs(self.covariances)), 1.0)
            ridge = jnp.sqrt(jnp.finfo(dtype).eps) * scale
        eye = ridge * jnp.eye(K, dtype=dtype)

        def per_component(Sigma):
            Sigma = Sigma + eye  # keep singular canonical / zero components invertible
            Sinv = jnp.linalg.inv(Sigma)
            _, logdet_S = jnp.linalg.slogdet(Sigma)

            def h(beta):
                return like_term(beta) - 0.5 * beta @ (Sinv @ beta)

            pot = laplace.fit(h, x0)  # MAP of the joint; precision already includes Sinv
            _, logdet_prec = jnp.linalg.slogdet(pot.precision)
            covar = jnp.linalg.inv(pot.precision)
            log_Z = pot.log_scale - 0.5 * logdet_S - 0.5 * logdet_prec
            return pot.mode, covar, log_Z

        post_mean, post_covar, log_Z = eqx.filter_vmap(per_component)(self.covariances)

        log_num = self.log_mix + log_Z
        log_marginal = logsumexp(log_num)
        resp = jnp.exp(log_num - log_marginal)
        warm = jnp.einsum("g,gk->k", resp, post_mean)

        return SERPosterior(resp=resp, mean=post_mean, covar=post_covar), log_marginal, warm

    def kl(self, posterior: SERPosterior) -> Array:
        """``KL( q(beta | variant) || prior )`` for one variant (augmented-model form).

        Treating the mixture component as a latent ``z`` with ``q(z=g) = resp_g`` and
        ``p(z=g) = w_g``, the KL is exact:
        ``sum_g resp_g [ log(resp_g / w_g) + KL(N(mean_g, covar_g) || N(0, Sigma_g)) ]``.
        """
        log_w = self.log_mix

        def per_component(r, mean_g, covar_g, Sigma_g, logw_g):
            kl_beta = _kl_mvn(mean_g, covar_g, Sigma_g)
            kl_z = jnp.log(r) - logw_g
            return jnp.where(r > 0, r * (kl_z + kl_beta), 0.0)

        terms = eqx.filter_vmap(per_component)(
            posterior.resp, posterior.mean, posterior.covar, self.covariances, log_w
        )
        return jnp.sum(terms)

    def update(self, alpha: Array, posterior: SERPosterior) -> "Prior":
        """Empirical-Bayes (EM M-step) update of the mixture from one effect's SER.

        Args:
            alpha: inclusion probabilities over variants (shape ``(M,)``).
            posterior: batched :class:`SERPosterior` over variants (leading ``M`` axis).

        Returns:
            A new :class:`Prior` with updated weights and covariances. Each variant's
            contribution is weighted by ``alpha_m`` (its inclusion probability) times the
            component responsibility ``r_{m,g}``.
        """
        # weight_{m,g} = alpha_m * resp_{m,g}
        w = alpha[:, None] * posterior.resp  # (M, G)
        comp_weight = w.sum(axis=0)  # (G,)

        # E[bb^T | m, g] = covar_{m,g} + mean_{m,g} mean_{m,g}^T
        second = posterior.covar + jnp.einsum(
            "mgk,mgl->mgkl", posterior.mean, posterior.mean
        )  # (M, G, K, K)

        # responsibility-weighted average 2nd moment per component; components with
        # negligible weight keep their previous covariance (avoids 0/0 -> NaN).
        eps = 1e-12
        K = self.covariances.shape[-1]
        weighted = jnp.einsum("mg,mgkl->gkl", w, second)
        denom = jnp.maximum(comp_weight, eps)[:, None, None]
        updated = weighted / denom
        keep = (comp_weight > eps)[:, None, None]
        Sigma = jnp.where(keep, updated, self.covariances)
        Sigma = Sigma + 1e-8 * jnp.eye(K, dtype=self.covariances.dtype)  # ridge to stay PD

        total = jnp.maximum(comp_weight.sum(), eps)
        new_log_weights = jnp.log(jnp.clip(comp_weight / total, eps, None))
        return Prior(log_weights=new_log_weights, covariances=Sigma)


class PointwiseLikelihood(AbstractLikelihood):
    """A factorised likelihood ``log p(Y | eta) = sum_i factor(y_i, eta_i, theta)``.

    Data is bound at construction (the ``y`` field). ``factor`` is stored statically and
    applied elementwise; ``theta`` holds any shared parameters (e.g. residual variance)
    as a pytree field so they are differentiable and EB-updatable.

    Because the density factorises, this subclass replaces two
    :class:`AbstractLikelihood` defaults with exact / cheaper forms: the expected
    log-density under a Gaussian belief ``eta ~ N(mean, var)`` is elementwise
    Gauss-Hermite quadrature (exact for a Gaussian likelihood, not just second-order), and
    the Hessian is diagonal so :meth:`hessian_diagonal` is the elementwise second
    derivative rather than the base's dense ``N x N`` build.
    """

    y: Array
    theta: dict
    factor: Callable = eqx.field(static=True)
    gh_nodes: Array
    gh_weights: Array
    update_fn: Callable | None = eqx.field(static=True, default=None)

    def log_density(self, eta: Array) -> Array:
        return jnp.sum(self.factor(self.y, eta, self.theta))

    def hessian_diagonal(self, eta: Array) -> Array:
        """Elementwise second derivative ``d^2 factor_i / d eta_i^2`` (shape of ``eta``).

        The factor is applied per observation, so ``Hessian(log_density)`` is diagonal and
        two elementwise autodiff passes recover it in ``O(N)`` -- no dense matrix. (Unused
        by the exact-quadrature :meth:`expected_log_density`; provided so the base's delta
        method / any external caller stays cheap rather than falling back to ``O(N^2)``.)
        """
        d_factor = jax.grad(lambda e: jnp.sum(self.factor(self.y, e, self.theta)))
        return jax.grad(lambda e: jnp.sum(d_factor(e)))(eta)

    def updated(self, eta_mean: Array, eta_var: Array) -> "PointwiseLikelihood":
        """Empirical-Bayes update of ``theta`` from the current latent belief.

        Returns ``self`` unchanged when the likelihood declares no ``update_fn``.
        """
        if self.update_fn is None:
            return self
        new_theta = self.update_fn(self.y, eta_mean, eta_var, self.theta)
        return eqx.tree_at(lambda lik: lik.theta, self, new_theta)

    def expected_log_density(self, mean: Array, var: Array) -> Array:
        """``sum_i E_{eta_i ~ N(mean_i, var_i)}[ factor(y_i, eta_i, theta) ]``.

        Uses ``E_{x~N(m,v)} g(x) = (1/sqrt(pi)) sum_q w_q g(m + sqrt(2v) t_q)`` with
        physicists' Gauss-Hermite nodes ``t_q`` / weights ``w_q``.

        ``var`` is clamped at zero before the ``sqrt``: the SER's effect remove/add-back
        bookkeeping can leave a tiny negative variance from catastrophic cancellation, and
        (unlike the Gaussian's analytic form) ``sqrt(2v)`` would turn that into ``nan`` and
        poison the whole quadrature. The Gaussian ELBO path clamps the same way.
        """
        scale = jnp.sqrt(2.0 * jnp.maximum(var, 0.0))

        def at_node(t, w):
            eta = mean + scale * t
            return w * self.factor(self.y, eta, self.theta)

        contribs = jax.vmap(at_node)(self.gh_nodes, self.gh_weights)  # (Q, ...)
        expected = contribs.sum(axis=0) / jnp.sqrt(jnp.pi)
        return jnp.sum(expected)


class Likelihood:
    """A :class:`PointwiseLikelihood` factory wrapping an elementwise log-density factor.

    Produced by the :func:`likelihood` decorator. Calling the instance binds the data
    and shared parameters, returning a ready :class:`PointwiseLikelihood`::

        lik = gaussian(y, sigma2=1.0)

    Unlike a bare closure, the wrapped ``factor`` stays reachable as the :attr:`factor`
    attribute (as does the original function via ``__wrapped__``), so callers can inspect
    or reuse the elementwise log-density directly. The Gauss-Hermite quadrature rule is
    precomputed once at construction and shared by every built likelihood.

    Attributes:
        factor: the wrapped elementwise log-density ``factor(y, eta, theta)``.
        update_fn: optional EB M-step ``(y, eta_mean, eta_var, theta) -> new_theta``.
        n_quad: number of Gauss-Hermite nodes.
        gh_nodes, gh_weights: the precomputed quadrature rule.
    """

    def __init__(
        self, factor: Callable, *, n_quad: int = 32, update_fn: Callable | None = None
    ):
        nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
        self.factor = factor
        self.update_fn = update_fn
        self.n_quad = n_quad
        self.gh_nodes = jnp.asarray(nodes)
        self.gh_weights = jnp.asarray(weights)
        functools.update_wrapper(self, factor)

    def __call__(self, y, **theta) -> PointwiseLikelihood:
        return PointwiseLikelihood(
            y=jnp.asarray(y),
            theta={k: jnp.asarray(v) for k, v in theta.items()},
            factor=self.factor,
            gh_nodes=self.gh_nodes,
            gh_weights=self.gh_weights,
            update_fn=self.update_fn,
        )


def likelihood(
    factor: Callable | None = None,
    *,
    n_quad: int = 32,
    update_fn: Callable | None = None,
) -> Callable:
    """Decorator turning an elementwise log-density into a :class:`Likelihood` factory.

    Apply it directly to a ``factor(y, eta, theta) -> per-element log p`` function, either
    bare or with keyword options::

        @likelihood
        def bernoulli_logit(y, eta, theta):
            return y * eta - jnp.logaddexp(0.0, eta)

        @likelihood(update_fn=_gaussian_update)
        def gaussian(y, eta, theta):
            ...

    The result is a callable :class:`Likelihood`: ``gaussian(y, sigma2=1.0)`` binds the
    data and parameters into a :class:`PointwiseLikelihood`, while ``gaussian.factor``
    still exposes the original elementwise function. Calling with a factor positionally
    (``likelihood(_factor, update_fn=...)``) also works, so it doubles as a plain wrapper.

    ``update_fn(y, eta_mean, eta_var, theta) -> new_theta`` (optional) is the empirical-
    Bayes M-step for the shared parameters ``theta``; ``n_quad`` (default 32) sets the
    number of Gauss-Hermite nodes for ``expected_log_density``.
    """
    if factor is None:
        # Parameterized form: @likelihood(update_fn=...) -> decorator awaiting the factor.
        return functools.partial(likelihood, n_quad=n_quad, update_fn=update_fn)
    return Likelihood(factor, n_quad=n_quad, update_fn=update_fn)


class Model(eqx.Module):
    """A fully specified generalized-SuSiE model: the four orthogonal axes.

    Attributes:
        design: the :class:`~lasusie.design.DesignMap` (`eta = X(B)`).
        likelihood: the observation likelihood (bound to its data).
        prior: the mixture-of-Gaussians effect prior.
        log_pi: log inclusion weights over variants (shape ``(M,)``; uniform by default).
        covariates: optional fixed-effect covariate offset
            (:class:`~lasusie.covariates.Covariates`); ``None`` means no covariates, in which
            case the latent predictor is ``eta = design(B)`` unchanged.
    """

    design: DesignMap
    likelihood: eqx.Module
    prior: Prior
    log_pi: Array
    covariates: Covariates | None = None

    @property
    def n_variants(self) -> int:
        return self.log_pi.shape[0]
