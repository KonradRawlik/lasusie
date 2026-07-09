from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array, tree_util


def damped_newton(
    f: Callable[[Array], Array],
    x0: Array,
    num_steps: int = 8,
    num_halvings: int = 30,
) -> Array:
    """Maximise a smooth concave ``f: R^d -> R`` by a few damped-Newton steps.

    Each step takes the Newton direction ``d = -H^{-1} g`` but backtracks the step length
    along a geometric ladder ``1, 1/2, 1/4, ...`` and keeps the length that maximises ``f``
    (non-finite values treated as ``-inf``), staying put if none improve. The full step is
    accepted whenever it is finite and best, so for a quadratic ``f`` a single undamped step
    is still exact; the damping only kicks in for nonlinear objectives where an undamped step
    could overshoot into an ``inf``/``nan`` region (e.g. ``exp(eta)`` overflow for log-link
    likelihoods). The ladder is a fixed unroll (no data-dependent ``while``), so the routine
    stays ``jit``/``vmap`` friendly. Runs on low-dimensional parameter vectors (covariate
    coefficients, ordinal cutpoints), so the per-step Hessian is cheap.
    """
    grad = jax.grad(f)
    hess = jax.hessian(f)
    d = x0.shape[0]
    # Keep every derived array in the warm-start dtype so the scan carry stays a single
    # precision (a config-default float64 ladder/eye against a float32 x0 would upcast it).
    ladder = (0.5 ** jnp.arange(num_halvings)).astype(x0.dtype)  # [1, 1/2, 1/4, ...]

    def step(x, _):
        g = grad(x)
        H = hess(x)
        # Ridge toward negative-definiteness (H is neg. def. at a max) for conditioning.
        direction = -jnp.linalg.solve(H - 1e-8 * jnp.eye(d, dtype=x0.dtype), g)
        f0 = f(x)
        candidates = x + ladder[:, None] * direction  # (num_halvings, d)
        vals = jax.vmap(f)(candidates)
        vals = jnp.where(jnp.isfinite(vals), vals, -jnp.inf)
        best = jnp.argmax(vals)
        # Accept the best step length only if it strictly improves on staying put.
        x_new = jnp.where(vals[best] > f0, candidates[best], x)
        return x_new, None

    x, _ = jax.lax.scan(step, x0, None, length=num_steps)
    return x


class MaybeIndexable:
    def __init__(self, x):
        self.x = x

    def __getitem__(self, i):
        return None if self.x is None else self.x[i]

def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)