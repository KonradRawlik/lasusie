"""Top-level entry point: run generalized SuSiE and summarise the result.

``finemap`` runs the IBSS loop and builds credible sets with a purity filter (a lean port
of sushie's ``make_cs``).
"""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

from .ibss import ibss
from .model import Model


class CredibleSet(NamedTuple):
    """A single credible set.

    Attributes:
        effect: the effect index ``l`` it came from.
        variants: variant indices in the set (ordered by descending alpha).
        coverage: total posterior inclusion mass captured (``sum alpha``).
        purity: minimum absolute pairwise genotype correlation among the variants.
        kept: whether the set passed the purity threshold.
    """

    effect: int
    variants: np.ndarray
    coverage: float
    purity: float
    kept: bool


class FineMapResult(NamedTuple):
    """Result of :func:`finemap`."""

    pip: np.ndarray
    alpha: np.ndarray
    credible_sets: list
    elbo: float
    elbo_history: list
    converged: bool
    iterations: int
    covariate_coef: np.ndarray | None  # fitted covariate coefficients gamma (None if no covariates)
    posterior: object  # the raw SuSiEResult


def _credible_sets(design, alpha, coverage, purity):
    L = alpha.shape[0]
    sets = []
    for l in range(L):
        a = alpha[l]
        order = jnp.argsort(a)[::-1]
        csum = jnp.cumsum(a[order])
        # smallest prefix reaching `coverage`
        n_below = int(jnp.sum(csum < coverage))
        n = min(n_below + 1, a.shape[0])
        idx = order[:n]
        pur = float(design.min_abs_corr(idx))
        sets.append(
            CredibleSet(
                effect=l,
                variants=np.asarray(idx),
                coverage=float(csum[n - 1]),
                purity=pur,
                kept=pur >= purity,
            )
        )
    return sets


def finemap(
    model: Model,
    L: int = 10,
    coverage: float = 0.95,
    purity: float = 0.5,
    dtype: DTypeLike | None = None,
    **ibss_kwargs: object,
) -> FineMapResult:
    """Fine-map with generalized SuSiE.

    Args:
        model: a fully specified :class:`~lasusie.model.Model`.
        L: number of single effects.
        coverage: target coverage for each credible set.
        purity: minimum purity (min abs correlation) to keep a credible set.
        dtype: floating precision for the run (see :func:`~lasusie.ibss.ibss`). ``None``
            infers it from the model's arrays; pass e.g. ``jnp.float32`` to force it. 64-bit
            precision additionally requires ``jax.config.update("jax_enable_x64", True)``.
        **ibss_kwargs: forwarded to :func:`~lasusie.ibss.ibss` (e.g. ``max_iter``,
            ``tol``, ``update_prior``, ``update_likelihood``, ``propagate_variance``).
    """
    res = ibss(model, L=L, dtype=dtype, **ibss_kwargs)
    sets = _credible_sets(model.design, res.alpha, coverage, purity)
    covariate_coef = (
        None if res.covariates is None else np.asarray(res.covariates.gamma)
    )
    return FineMapResult(
        pip=np.asarray(res.pip),
        alpha=np.asarray(res.alpha),
        credible_sets=sets,
        elbo=res.elbo,
        elbo_history=res.elbo_history,
        converged=res.converged,
        iterations=res.iterations,
        covariate_coef=covariate_coef,
        posterior=res,
    )
