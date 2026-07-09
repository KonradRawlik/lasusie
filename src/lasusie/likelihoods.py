"""Built-in likelihoods, defined as minimal elementwise log-density factors.

Each factor ``(y, eta, theta)`` returns the per-observation log density and is wrapped by
the :func:`lasusie.model.likelihood` decorator, which supplies autodiff derivatives and
the Gauss-Hermite ``expected_log_density`` used by the variance-propagating SER. The
decorated names below are :class:`~lasusie.model.Likelihood` factories -- call them to
bind data (``gaussian(y, sigma2=1.0)``); the underlying factor stays on the ``.factor``
attribute.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.scipy.special import gammaln, log_ndtr

from .model import AbstractLikelihood, PointwiseLikelihood, likelihood
from .util import damped_newton


def _gaussian_update(y, eta_mean, eta_var, theta):
    # EB residual variance: E[(y - eta)^2] averaged over observations.
    erss = jnp.sum((y - eta_mean) ** 2) + jnp.sum(eta_var)
    return {"sigma2": erss / y.size}


@likelihood(update_fn=_gaussian_update)
def gaussian(y, eta, theta):
    """Gaussian log-density, identity link, with EB-updatable residual variance."""
    sigma2 = theta["sigma2"]
    return -0.5 * jnp.log(2.0 * jnp.pi * sigma2) - 0.5 * (y - eta) ** 2 / sigma2


@likelihood
def bernoulli_logit(y, eta, theta):
    """Bernoulli log-density, logit link: ``y*eta - softplus(eta)`` for ``y in {0,1}``."""
    return y * eta - jnp.logaddexp(0.0, eta)


@likelihood
def poisson_log(y, eta, theta):
    """Poisson log-density, log link: ``y*eta - exp(eta) - log(y!)``."""
    return y * eta - jnp.exp(eta) - gammaln(y + 1.0)


@likelihood
def neg_binomial_log(y, eta, theta):
    """Negative-binomial log-density, log link (NB2: ``var = mu + mu^2/r``).

    Mean ``mu = exp(eta)``; ``r = theta["r"]`` (default 1.0) is the size/dispersion --
    ``r -> inf`` recovers Poisson, smaller ``r`` means more overdispersion, the usual
    fit for RNA-seq / eQTL counts. Call ``neg_binomial_log(y, r=...)``. ``r`` is fixed
    (user-supplied) in v0 -- it has no closed-form EB update (see module note)."""
    r = theta.get("r", 1.0)
    # log(r + exp(eta)) via logaddexp keeps the factor finite (it grows only linearly in
    # eta), so a large offset variance can't overflow the outer Gauss-Hermite quadrature.
    log_rpm = jnp.logaddexp(jnp.log(r), eta)
    return (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        + r * (jnp.log(r) - log_rpm)
        + y * (eta - log_rpm)
    )


@likelihood
def student_t(y, eta, theta):
    """Student-t log-density, identity link (robust/heavy-tailed regression).

    Location ``eta``, scale^2 ``theta["sigma2"]`` (default 1.0), d.o.f. ``theta["nu"]``
    (default 4.0). ``nu -> inf`` recovers the Gaussian. Call
    ``student_t(y, nu=..., sigma2=...)``. Both are fixed in v0."""
    nu = theta.get("nu", 4.0)
    s2 = theta.get("sigma2", 1.0)
    z2 = (y - eta) ** 2 / s2
    return (
        gammaln(0.5 * (nu + 1.0))
        - gammaln(0.5 * nu)
        - 0.5 * jnp.log(nu * jnp.pi * s2)
        - 0.5 * (nu + 1.0) * jnp.log1p(z2 / nu)
    )


@likelihood
def gamma_log(y, eta, theta):
    """Gamma log-density, log link, for positive continuous responses.

    Mean ``mu = exp(eta)``, shape ``k = theta["shape"]`` (default 1.0, which is the
    exponential distribution); rate ``= k/mu``. Call ``gamma_log(y, shape=...)``.
    ``k`` is fixed in v0."""
    k = theta.get("shape", 1.0)
    # rate = k / mu = k * exp(-eta); log(rate) = log(k) - eta.
    return (
        k * (jnp.log(k) - eta)
        - gammaln(k)
        + (k - 1.0) * jnp.log(y)
        - k * jnp.exp(-eta) * y
    )


def _beta_binomial_factor(data, eta, theta):
    successes = data[..., 0]
    trials = data[..., 1]
    s = theta.get("s", 10.0)  # concentration; s -> inf recovers the binomial
    p = jax.nn.sigmoid(eta)
    a = p * s
    b = (1.0 - p) * s
    log_choose = (
        gammaln(trials + 1.0) - gammaln(successes + 1.0) - gammaln(trials - successes + 1.0)
    )
    return (
        log_choose
        + gammaln(successes + a)
        + gammaln(trials - successes + b)
        - gammaln(trials + s)
        - (gammaln(a) + gammaln(b) - gammaln(s))
    )


_beta_binomial = likelihood(_beta_binomial_factor)


def beta_binomial_logit(y, n, s=10.0):
    """Beta-binomial log-density, logit link (overdispersed proportions).

    For ``y`` successes out of ``n`` trials with mean probability ``sigmoid(eta)`` and
    concentration ``s`` (larger ``s`` -> less overdispersion; ``s -> inf`` is the
    binomial). Typical for allele-specific expression / methylation counts. ``y`` and
    ``n`` are 1-D of length N (``n`` may be scalar); ``s`` is fixed in v0."""
    y = jnp.asarray(y).astype(float).reshape(-1, 1)
    n = jnp.broadcast_to(jnp.asarray(n).astype(float).reshape(-1, 1), y.shape)
    data = jnp.stack([y, n], axis=-1)  # (N, 1, 2)
    return _beta_binomial(data, s=s)


# --- Ordinal (cumulative-link / proportional-odds) with EB-estimated cutpoints ---------
#
# For C ordered categories y in {0, ..., C-1} with ordered cutpoints
# alpha_1 < ... < alpha_{C-1} (alpha_0 = -inf, alpha_C = +inf):
#
#     P(Y = c | eta) = F(alpha_{c+1} - eta) - F(alpha_c - eta),
#
# with F the logistic CDF (ordered logit) or normal CDF (ordered probit). A single latent
# predictor eta shifts every category boundary -- the proportional-odds assumption -- which
# maps exactly onto lasusie's one-dimensional eta, so this is an ordinary pointwise
# likelihood. The log-density is log-concave in eta, so (unlike the count models) the
# per-variant fit is well behaved with no divergent-MLE drama.
#
# The C-1 cutpoints are shared nuisance parameters, EB-estimated between IBSS sweeps via
# ``update_fn`` (like ``gaussian``'s residual variance). They must stay ordered, so they are
# carried unconstrained as ``cut0 = alpha_1`` plus ``log_deltas = log(alpha_{c+1} - alpha_c)``
# and reconstructed by an exp-cumsum. NOTE: the cutpoints absorb the outcome's baseline level,
# so an ordinal model should NOT also carry a covariate intercept (they are confounded).


def _log1mexp(z: Array) -> Array:
    """Numerically stable ``log(1 - exp(z))`` for ``z <= 0``."""
    return jnp.where(
        z < -0.6931471805599453,  # -log 2
        jnp.log1p(-jnp.exp(z)),
        jnp.log(-jnp.expm1(z)),
    )


def _cutpoints(theta) -> Array:
    """Ordered cutpoints ``(C-1,)`` from the unconstrained ``(cut0, log_deltas)``."""
    cut0 = theta["cut0"]
    increments = jnp.cumsum(jnp.exp(theta["log_deltas"]))  # (C-2,), strictly positive
    return jnp.concatenate([jnp.atleast_1d(cut0), cut0 + increments])


# (log CDF, log survival) of the baseline error: logistic -> logit, standard normal -> probit.
_ORDINAL_LINKS = {
    "logit": (lambda x: -jax.nn.softplus(-x), lambda x: -jax.nn.softplus(x)),
    "probit": (log_ndtr, lambda x: log_ndtr(-x)),
}


def _ordinal_factor(log_cdf, log_sf, C):
    def factor(y, eta, theta):
        cuts = _cutpoints(theta)  # (C-1,), alpha_1 < ... < alpha_{C-1}
        yi = jnp.round(y).astype(jnp.int32)  # category in {0, ..., C-1}
        # Clip both gathers in-range so no +/-inf boundary ever reaches log_cdf/log_sf; the
        # out-of-range side is masked out below (keeps values and gradients finite).
        upper = cuts[jnp.clip(yi, 0, C - 2)] - eta      # alpha_{yi+1} - eta
        lower = cuts[jnp.clip(yi - 1, 0, C - 2)] - eta  # alpha_{yi}   - eta
        lu = log_cdf(upper)
        is_mid = (yi != 0) & (yi != C - 1)
        # Sanitise the log1mexp argument for boundary categories (where upper==lower would
        # give log1mexp(0) = -inf and an infinite gradient) -- their value is masked anyway.
        z = jnp.where(is_mid, log_cdf(lower) - lu, -1.0)
        log_mid = lu + _log1mexp(z)  # log(F(upper) - F(lower))
        return jnp.where(
            yi == 0,
            lu,  # P(Y=0) = F(alpha_1 - eta)
            jnp.where(yi == C - 1, log_sf(lower), log_mid),  # P(Y=C-1) = 1 - F(alpha_{C-1}-eta)
        )

    return factor


def _ordinal_update(log_cdf, log_sf, C, n_quad=20):
    """EB M-step: Newton-maximise the expected log-likelihood over the cutpoints."""
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    nodes = jnp.asarray(nodes)
    weights = jnp.asarray(weights)
    factor = _ordinal_factor(log_cdf, log_sf, C)

    def update(y, eta_mean, eta_var, theta):
        flat0, unflatten = ravel_pytree(theta)
        scale = jnp.sqrt(2.0 * jnp.maximum(eta_var, 0.0))

        def expected_ll(flat):  # sum_i E_{eta~N(mean,var)} log P(y_i | eta, cutpoints)
            th = unflatten(flat)
            at = jax.vmap(lambda t, w: w * factor(y, eta_mean + scale * t, th))(nodes, weights)
            return jnp.sum(at.sum(axis=0) / jnp.sqrt(jnp.pi))

        return unflatten(damped_newton(expected_ll, flat0))

    return update


def _ordinal(y, n_categories, link, cutpoints) -> PointwiseLikelihood:
    C = int(n_categories)
    if C < 2:
        raise ValueError("ordinal likelihood needs n_categories >= 2")
    log_cdf, log_sf = _ORDINAL_LINKS[link]
    if cutpoints is None:
        init = jnp.zeros(1) if C == 2 else jnp.linspace(-1.0, 1.0, C - 1)
    else:
        init = jnp.asarray(cutpoints, dtype=float)
    cut0 = init[0]
    log_deltas = jnp.zeros(0) if C == 2 else jnp.log(jnp.diff(init))
    lik = likelihood(
        _ordinal_factor(log_cdf, log_sf, C),
        update_fn=_ordinal_update(log_cdf, log_sf, C),
    )
    return lik(jnp.asarray(y).astype(float).reshape(-1, 1), cut0=cut0, log_deltas=log_deltas)


def ordinal_logit(y, n_categories, cutpoints=None) -> PointwiseLikelihood:
    """Ordered-logit (proportional-odds) likelihood for ordinal ``y`` in ``{0..C-1}``.

    ``n_categories`` is ``C``; the ``C-1`` cutpoints are EB-estimated between sweeps (pass
    ``cutpoints`` to fix/initialise them, else evenly spaced). The cutpoints absorb the
    outcome baseline, so do not also add a covariate intercept."""
    return _ordinal(y, n_categories, "logit", cutpoints)


def ordinal_probit(y, n_categories, cutpoints=None) -> PointwiseLikelihood:
    """Ordered-probit likelihood (normal-CDF link); see :func:`ordinal_logit`."""
    return _ordinal(y, n_categories, "probit", cutpoints)


class MVNResidual(AbstractLikelihood):
    """Multi-phenotype Gaussian likelihood: ``y_n ~ N(eta_n, Sigma_e)`` per individual.

    The residual covariance ``Sigma_e`` (``K x K``, across phenotypes on the same person)
    is the multi-phenotype coupling: it makes the SER precision ``Lambda_m`` dense (its
    off-diagonal comes from ``Sigma_e^{-1}``), on top of any prior coupling.

    The expectation under a Gaussian offset belief is exact; here we use the diagonal of
    the offset covariance (``var`` has the latent shape ``(N, K)``) -- the same mean-field
    approximation of the cross-context offset covariance that the plan defers.

    Attributes:
        y: phenotypes (shape ``(N, K)``).
        Sigma_e: residual covariance (shape ``(K, K)``), EB-updatable.
    """

    y: Array
    Sigma_e: Array

    def log_density(self, eta: Array) -> Array:
        r = self.y - eta  # (N, K)
        prec = jnp.linalg.inv(self.Sigma_e)
        _, logdet = jnp.linalg.slogdet(self.Sigma_e)
        K = self.Sigma_e.shape[0]
        quad = jnp.einsum("nk,kl,nl->n", r, prec, r)
        return jnp.sum(-0.5 * (K * jnp.log(2 * jnp.pi) + logdet + quad))

    def expected_log_density(self, mean: Array, var: Array) -> Array:
        # E_{eta_n ~ N(mean_n, diag(var_n))} log N(y_n; eta_n, Sigma_e)
        #   = log N(y_n; mean_n, Sigma_e) - 0.5 tr(Sigma_e^{-1} diag(var_n))
        prec = jnp.linalg.inv(self.Sigma_e)
        base = self.log_density(mean)  # uses mean as eta
        trace_term = 0.5 * jnp.sum(var * jnp.diag(prec))  # sum_n tr(prec diag(var_n))
        return base - trace_term

    def updated(self, eta_mean: Array, eta_var: Array) -> "MVNResidual":
        r = self.y - eta_mean  # (N, K)
        N = r.shape[0]
        Sigma = (r.T @ r + jnp.diag(jnp.sum(eta_var, axis=0))) / N
        return MVNResidual(y=self.y, Sigma_e=Sigma)


def mvn_resid(y, resid_cov=None) -> MVNResidual:
    """Construct a :class:`MVNResidual` likelihood for phenotypes ``y`` (``N x K``)."""
    y = jnp.asarray(y)
    K = y.shape[1]
    Sigma_e = jnp.eye(K) if resid_cov is None else jnp.asarray(resid_cov)
    return MVNResidual(y=y, Sigma_e=Sigma_e)


class CoxPH(AbstractLikelihood):
    """Cox proportional-hazards partial likelihood -- a *composite* likelihood.

    ``log PL(eta) = sum_{i: event} [ eta_i - log sum_{j in risk(t_i)} exp(eta_j) ]``.
    The risk-set denominators couple all samples, so this is not factorised. Preprocessing
    (:meth:`bind`) sorts individuals by descending time *once*, turning the risk-set sums
    into a cumulative log-sum-exp (O(N) per evaluation). The sort is data-derived and
    ``eta``-independent, so it is stored as a field and treated as a constant by autodiff.

    Only :meth:`log_density` and :meth:`hessian_diagonal` are implemented; the delta-method
    :meth:`~lasusie.model.AbstractLikelihood.expected_log_density` and the identity
    :meth:`~lasusie.model.AbstractLikelihood.updated` are inherited from
    :class:`~lasusie.model.AbstractLikelihood`. Offset-variance propagation therefore
    accounts for each individual's *marginal* offset variance but not cross-individual
    covariance (a true N-dimensional integral, deferred). Supplying the structured
    :meth:`hessian_diagonal` makes that correction cost ``O(N)`` via risk-set cumulative
    sums instead of the base default's ``O(N^2)`` dense-Hessian build. The mode-finding
    objective ignores the correction (only its value is used), so it does not affect where
    the Laplace mode is found. Ties are handled by the Breslow approximation (exact when
    event times are distinct).

    Attributes:
        order: permutation sorting individuals by descending time.
        events_sorted: event indicators reordered by ``order``.
    """

    order: Array
    events_sorted: Array

    @classmethod
    def bind(cls, times: Array, events: Array) -> "CoxPH":
        """Preprocess survival data once: sort by descending time."""
        times = jnp.asarray(times)
        events = jnp.asarray(events)
        order = jnp.argsort(times)[::-1]
        return cls(order=order, events_sorted=events[order])

    def log_density(self, eta: Array) -> Array:
        e = eta.reshape(-1)[self.order]
        log_risk = jax.lax.cumlogsumexp(e)  # risk set = positions 0..p (>= time)
        return jnp.sum(self.events_sorted * (e - log_risk))

    def hessian_diagonal(self, eta: Array) -> Array:
        """Diagonal of ``Hessian(log_density)(eta)``, exact, in ``O(N)``.

        The full Hessian ``H = -sum_{i: event} (diag(P_i) - P_i P_i^T)`` is dense (the
        risk-set denominators couple all samples), but the delta-method correction needs
        only its diagonal

            H_nn = -sum_{i: event, t_i <= t_n} P_in (1 - P_in),   P_in = exp(eta_n) / S_i,

        where ``S_i`` is the risk-set denominator at event ``i``. Splitting ``P(1 - P)``
        into ``sum P`` and ``sum P^2`` turns each over-events sum into a *suffix* (reverse
        cumulative) log-sum-exp over the descending-time order -- the mirror image of the
        forward ``cumlogsumexp`` used for the value. So the whole diagonal costs ``O(N)``
        instead of the ``O(N^2)`` of building ``H`` and calling ``jnp.diagonal``. Working in
        log-space keeps every intermediate bounded: ``S_i >= exp(eta_n)`` for ``i`` in the
        suffix, so each ``P_in <= 1`` and the exponents below never overflow.
        """
        e = eta.reshape(-1)[self.order]  # descending-time order
        log_risk = jax.lax.cumlogsumexp(e)  # log S_i at each position
        # For the sample at position k, the events whose risk set contains it are exactly
        # those at positions i >= k (later position == not-larger time). Accumulate the
        # event-masked 1/S_i and 1/S_i^2 as reverse cumulative log-sum-exps; a non-event
        # contributes log 0 = -inf, i.e. nothing.
        mask = jnp.where(self.events_sorted > 0, 0.0, -jnp.inf)
        log_A = jax.lax.cumlogsumexp(mask - log_risk, reverse=True)  # log sum 1/S_i
        log_B = jax.lax.cumlogsumexp(mask - 2.0 * log_risk, reverse=True)  # log sum 1/S_i^2
        diag_sorted = jnp.exp(2.0 * e + log_B) - jnp.exp(e + log_A)  # sum P^2 - sum P <= 0
        return jnp.zeros_like(e).at[self.order].set(diag_sorted).reshape(eta.shape)


def cox(times, events) -> CoxPH:
    """Construct a :class:`CoxPH` composite likelihood from survival data."""
    return CoxPH.bind(times, events)


# --- Parametric accelerated failure-time (AFT) survival models --------------------
#
# AFT posits ``log T = eta + scale * W`` for a baseline error ``W``. Unlike Cox (whose
# risk sets couple all samples into a composite likelihood), a *parametric* AFT model
# factorises over individuals with right-censoring folded in per observation, so it is
# an ordinary pointwise likelihood and gets Gauss-Hermite variance propagation for free.
#
# Writing ``z = (log t - eta) / scale``, an observation contributes its log-density
# ``log f(t) = -log t - log(scale) + log f_W(z)`` if the event was observed, or its log
# survival ``log S(t) = log S_W(z)`` if right-censored. Each baseline ``W`` below supplies
# ``(log f_W, log S_W)``; ``scale`` is fixed (user-supplied) in v0 -- with censoring it
# has no closed-form EB update (see the module note).


def _aft_factor(log_pdf_w, log_sf_w):
    """Build a pointwise AFT factor from a baseline error's ``(log f_W, log S_W)``."""

    def factor(data, eta, theta):
        log_t = data[..., 0]
        event = data[..., 1]  # 1 if the event was observed, 0 if right-censored
        scale = theta.get("scale", 1.0)
        z = (log_t - eta) / scale
        log_f = -log_t - jnp.log(scale) + log_pdf_w(z)
        log_s = log_sf_w(z)
        return event * log_f + (1.0 - event) * log_s

    return factor


# Baseline errors: standard normal -> log-normal AFT; standard (minimum) Gumbel ->
# Weibull AFT; standard logistic -> log-logistic AFT.
_aft_lognormal_lik = likelihood(
    _aft_factor(
        lambda z: -0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * z**2,
        lambda z: log_ndtr(-z),
    )
)
_aft_weibull_lik = likelihood(
    _aft_factor(lambda z: z - jnp.exp(z), lambda z: -jnp.exp(z))
)
_aft_loglogistic_lik = likelihood(
    _aft_factor(lambda z: -z - 2.0 * jax.nn.softplus(-z), lambda z: -jax.nn.softplus(z))
)


def _aft_data(times, events):
    t = jnp.asarray(times).astype(float).reshape(-1, 1)
    event = (
        jnp.ones_like(t) if events is None else jnp.asarray(events).astype(float).reshape(-1, 1)
    )
    return jnp.stack([jnp.log(t), event], axis=-1)  # (N, 1, 2)


def aft_lognormal(times, events=None, scale=1.0) -> PointwiseLikelihood:
    """Log-normal accelerated failure-time model (``log T ~ N(eta, scale^2)``).

    ``times > 0`` are event/censoring times; ``events`` is 1 for an observed event and
    0 for right-censoring (default: all observed). ``scale`` is fixed in v0."""
    return _aft_lognormal_lik(_aft_data(times, events), scale=scale)


def aft_weibull(times, events=None, scale=1.0) -> PointwiseLikelihood:
    """Weibull accelerated failure-time model (extreme-value errors).

    Same interface as :func:`aft_lognormal`; the Weibull shape is ``1/scale``, so it also
    happens to be a proportional-hazards model. ``scale`` is fixed in v0."""
    return _aft_weibull_lik(_aft_data(times, events), scale=scale)


def aft_loglogistic(times, events=None, scale=1.0) -> PointwiseLikelihood:
    """Log-logistic accelerated failure-time model (logistic errors).

    Same interface as :func:`aft_lognormal`; allows non-monotone (rise-then-fall) hazards.
    ``scale`` is fixed in v0."""
    return _aft_loglogistic_lik(_aft_data(times, events), scale=scale)
