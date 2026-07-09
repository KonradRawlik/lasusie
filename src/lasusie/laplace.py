"""Nested Laplace approximation of a single-effect likelihood in effect-space.

The single-effect regression needs, per candidate variant, a Gaussian approximation
of ``g(beta) = E_q[log p(Y | eta(beta))]`` as a function of the K-vector effect
``beta``.  Because ``beta`` is only K-dimensional (K = number of contexts, small),
the mode-find and curvature are cheap regardless of the sample size N or the
likelihood structure.

Mode-finding uses a *frozen-metric* (chord) iteration rather than full Newton: the
inverse Hessian is computed once at the (warm-started) initial point and reused as a
fixed preconditioner for a fixed number of gradient steps.  Each step is then only a
first-order gradient evaluation, and the fixed step count keeps the whole thing
``jit``/``vmap`` friendly (no data-dependent ``while`` loop).  This is safe here
because the chord method converges at a rate governed by how much the Hessian varies
between the initial point and the mode -- exactly the quantity the Laplace
approximation already assumes is small.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class GaussianPotential(NamedTuple):
    """Gaussian approximation of a log-density in effect-space, shared across modules.

    Represents ``f(beta) ~= log_scale - 0.5 (beta - mode)^T precision (beta - mode)``.

    Attributes:
        log_scale: the value of ``f`` at ``mode`` (a scalar).
        mode: the maximiser ``argmax f`` (shape ``(K,)``).
        precision: ``-Hessian f`` at ``mode`` (shape ``(K, K)``, positive definite
            at a maximum).
    """

    log_scale: Array
    mode: Array
    precision: Array


def fit(
    f: Callable[[Array], Array],
    x0: Array,
    num_steps: int = 8,
    mode_bound: float = 1e2,
    precision_floor: float = 1e-10,
) -> GaussianPotential:
    """Laplace-approximate a scalar log-density ``f: R^K -> R`` around its maximum.

    Args:
        f: the (concave near its mode) log-density to approximate.
        x0: warm-start init for the mode (shape ``(K,)``).
        num_steps: fixed number of frozen-metric ascent steps.
        mode_bound: the located mode is clipped to ``[-mode_bound, mode_bound]``.
        precision_floor: ridge added to the curvature (see below).

    Returns:
        A :class:`GaussianPotential` at the located mode.

    The ``mode_bound`` clip and ``precision_floor`` ridge guard against a *divergent*
    single-effect MLE. For a strictly concave likelihood (e.g. Gaussian) the mode is finite
    and well inside the bound and the curvature is large, so both guards are inert and the
    fit is exact as before. But a count / exponential-family likelihood
    (``poisson_log`` / ``neg_binomial_log`` / ``gamma_log``) is only *linear* in the tail, so
    a null (quasi-separated) variant has no interior maximum: the mode runs off to infinity
    and the curvature collapses to zero. Left unchecked, the runaway mode overflows to ``inf``
    (poisoning ``f`` on the next warm-started sweep) and the vanishing precision makes the
    downstream conjugate ``combine`` invert a singular matrix. Clipping the mode and flooring
    the precision keep such an uninformative variant finite and well-conditioned -- it simply
    contributes negligible evidence, the statistically correct outcome.
    """
    grad = jax.grad(f)
    K = x0.shape[0]
    # Match the warm-start dtype so the ridge never promotes the mode-finding arithmetic
    # (a float64 eye against a float32 x0 would upcast the scan carry and error out).
    ridge = precision_floor * jnp.eye(K, dtype=x0.dtype)

    # Precision at the init; the K x K inverse is trivial. The point of freezing it
    # is that we differentiate the (N-sample) likelihood at second order only ONCE. The
    # ridge also bounds the step when the warm-start curvature has collapsed to zero.
    precision0 = -jax.hessian(f)(x0)
    precond = jnp.linalg.inv(precision0 + ridge)

    def step(beta, _):
        # Ascent Newton step with the frozen metric: beta <- beta + P grad f(beta).
        return beta + precond @ grad(beta), None

    mode, _ = jax.lax.scan(step, x0, None, length=num_steps)
    mode = jnp.clip(mode, -mode_bound, mode_bound)

    # Recompute the curvature once at the (clipped) mode for an accurate precision.
    precision = -jax.hessian(f)(mode) + ridge
    return GaussianPotential(log_scale=f(mode), mode=mode, precision=precision)
