# Install

`lasusie` is a JAX implementation of generalized SuSiE fine-mapping. It is
managed with [uv](https://docs.astral.sh/uv/) and targets Python 3.11.

## From source

```bash
git clone <repo-url> lasusie
cd lasusie
uv sync
```

This installs the runtime dependencies (`jax`, `equinox`, `lineax`,
`optimistix`, `numpy`) into a local `.venv`.

!!! note "Apple Silicon"
    The default dependency set pins `jax-metal` for GPU acceleration on
    macOS. If you are on Linux or don't want the Metal backend, install a
    plain `jax` build instead and drop `jax-metal` from `pyproject.toml`.

## Development install

The `dev` dependency group adds `pytest`, `pandas`, and `jupyterlab`:

```bash
uv sync --group dev
uv run pytest
```

## Building the docs

Documentation is built with [MkDocs](https://www.mkdocs.org/) and the
Material theme. Install the `docs` group and serve locally:

```bash
uv sync --group docs
uv run mkdocs serve
```

This starts a live-reloading server at `http://127.0.0.1:8000`. To build a
static site into `site/`:

```bash
uv run mkdocs build
```

## Where to go next

- [Model](overview/model.md) — the class of models `lasusie` fits.
- [Algorithm](overview/algorithm.md) — how inference works, including the two levels
  of approximation.
- [Built-in likelihoods](overview/likelihoods.md) — the likelihoods shipped
  with `lasusie`.
- [Extending](extending/index.md) — how to add a new design, likelihood, or
  prior.
- [API Reference](api.md) — generated from docstrings.

## License

`lasusie` is released under the
[MIT License](https://github.com/KonradRawlik/lasusie/blob/main/LICENSE.md).

## Development note

[Claude Code](https://claude.com/claude-code) was used in aspects of the
development of this library.
