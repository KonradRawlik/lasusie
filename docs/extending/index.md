# Extending

`lasusie` fits models along three independent axes — **design**, **likelihood**,
**prior** (see [Model](../overview/model.md)). Adding a new model class means
implementing one of these axes; the other two, and the IBSS loop that ties
them together (see [Algorithm](../overview/algorithm.md)), are unaffected.

- [Adding a likelihood](likelihood.md) — the observation model \(\phi\)
  relating the latent predictor to the data. Most new use cases live here.
- [Adding a design](design.md) — the linear map from the effect array to the
  latent predictor(s).
- [Adding a prior](prior.md) — the mixture of sharing patterns for the
  effect-size vector.
