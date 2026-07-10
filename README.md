# lasusie

Generalized SuSiE fine-mapping in JAX.

`lasusie` is a JAX implementation of generalized SuSiE fine-mapping, factored
along three independent axes — design, likelihood, and prior — so the same
Iterative Bayesian Stepwise Selection (IBSS) algorithm applies across model
families.

## Documentation

Full documentation is at **https://konradrawlik.github.io/lasusie/**, including:

- [Model](https://konradrawlik.github.io/lasusie/overview/model/) — the class of models `lasusie` fits.
- [Algorithm](https://konradrawlik.github.io/lasusie/overview/algorithm/) — how inference works.
- [Built-in likelihoods](https://konradrawlik.github.io/lasusie/overview/likelihoods/) — the likelihoods shipped with `lasusie`.
- [Extending](https://konradrawlik.github.io/lasusie/extending/) — adding a design, likelihood, or prior.
- [API Reference](https://konradrawlik.github.io/lasusie/api/) — generated from docstrings.

## Install

`lasusie` is managed with [uv](https://docs.astral.sh/uv/) and targets Python 3.11.

```bash
git clone https://github.com/KonradRawlik/lasusie.git
cd lasusie
uv sync
```

See the [installation guide](https://konradrawlik.github.io/lasusie/) for the
`dev`, `docs`, and Apple Silicon (`jax-metal`) options.

## License

`lasusie` is released under the [MIT License](LICENSE.md).

## Development note

[Claude Code](https://claude.com/claude-code) was used in aspects of the
development of this library.
