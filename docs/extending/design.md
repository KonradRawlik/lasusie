# Adding a design

A design implements the linear map \(\eta = X(B)\) from the \(M \times K\)
effect array \(B\) to the latent predictor(s) (see
[Model](../overview/model.md#axis-1-design-how-effects-become-predictors)). The two
built-in designs — shared and block genotypes — are both configurations of
this one operator; a new design is needed only when the relationship between
effects and predictors isn't a plain matrix multiply against one of those two
layouts, e.g. genotype dosages that need an implied covariate adjustment
baked into the map, or a predictor built from more than one design matrix per
context.

A design needs to supply four operations:

1. **The forward map**, \(\mathrm{apply}(B) = X(B)\) — used to reconstitute
   the latent predictor from the current effect estimates.
2. **A single effect's contribution**, \(\mathrm{apply\_effect}(m, \beta)\)
   — the predictor contribution of one candidate variant \(m\) with effect
   size \(\beta \in \mathbb{R}^K\), equivalent to `apply` applied to a
   one-hot effect array but formed directly (this is what each SER evaluates
   many times per candidate variant, so it should avoid materializing the
   full \(B\)).
3. **The forward map on squared entries**, used for the latent predictor's
   second moment \(\mathbb{E}[\eta^2]\) under the variational posterior —
   the same linear structure as `apply`, but applied to squared design
   entries against the posterior's per-coordinate second moments rather than
   its means.
4. **Purity of a variant set** — the minimum absolute pairwise correlation
   among a set of candidate variants' columns, used to filter credible sets
   (see [Output](../overview/algorithm.md#output)). For a multi-context design this
   is a per-context computation, aggregated across contexts.

```python
import equinox as eqx
import jax.numpy as jnp
from jax import Array


class MyDesign(eqx.Module):
    # design-specific data, as pytree fields
    X: Array

    def apply(self, B: Array) -> Array:
        ...

    def apply_effect(self, m: Array, beta: Array) -> Array:
        ...

    def apply_sq(self, A: Array) -> Array:
        ...

    def min_abs_corr(self, idx: Array) -> Array:
        ...
```

- All four methods must be differentiable where `beta`/`B` appear — the SER
  takes gradients and Hessians through `apply_effect` when Laplace-fitting
  the single-effect potential (see
  [Algorithm](../overview/algorithm.md#approximation-1-nested-laplace-for-the-single-effect-potential)).
- `apply_sq` only needs to be correct for non-negative squared inputs; it
  does not need to be a generic linear map.
- `min_abs_corr` should return `1.0` for a singleton variant set (no pairwise
  correlation to compute).

Add the constructor to `lasusie.design` (and `__all__` in
[`lasusie/__init__.py`](../api.md) if it should be public). Nothing else in
the model needs to change — `Model(design=MyDesign(...), likelihood=...,
prior=..., log_pi=...)` works with any object implementing these four
methods.

## Testing a new design

At minimum, verify:

- `apply` and `apply_effect` agree: `apply_effect(m, beta)` matches
  `apply` on a one-hot effect array with `beta` at row `m`.
- `apply_sq` agrees with `apply` applied to squared inputs where the two
  should coincide (e.g. a binary design matrix).
- `min_abs_corr` returns `1.0` for a singleton set and matches a direct
  correlation computation for a small multi-variant set.

See `SharedDesign`/`BlockDesign` in [`lasusie/design.py`](../api.md) for
reference implementations to follow.
