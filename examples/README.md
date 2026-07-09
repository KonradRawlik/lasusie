# lasusie examples

Runnable tutorial notebooks for generalized SuSiE fine-mapping. Each notebook simulates
data with known causal variants and recovers them, so you can see the full workflow —
build a `Model`, call `finemap`, read PIPs and credible sets — end to end.

| notebook | what it covers |
|----------|----------------|
| [`01_standard_susie.ipynb`](01_standard_susie.ipynb)        | Classic univariate Gaussian SuSiE on a region with LD |
| [`02_survival_susie.ipynb`](02_survival_susie.ipynb)        | Survival outcomes with censoring — Cox PH and a parametric Weibull AFT |
| [`03_multiphenotype_susie.ipynb`](03_multiphenotype_susie.ipynb) | Multiple phenotypes: Gaussian (`mvn_resid`), non-Gaussian (binary), and a mixed (continuous + binary) likelihood |
| [`04_multiancestry_susie.ipynb`](04_multiancestry_susie.ipynb)   | Multi-ancestry fine-mapping across ancestries with different LD (`BlockDesign`) |

Each notebook is self-contained and reads top to bottom in a couple of minutes.

## Running them

The tutorial dependencies (JupyterLab, matplotlib, seaborn) live in the `examples`
dependency group:

```bash
uv sync --group examples
uv run --group examples jupyter lab   # then open examples/*.ipynb
```

### CPU vs. Metal backend

This project pins `jax-metal`, which on Apple Silicon becomes the default JAX backend.
The Metal backend can choke on some ops these notebooks use, so run them on the **CPU
backend** by setting `JAX_PLATFORMS=cpu`:

```bash
JAX_PLATFORMS=cpu uv run --group examples jupyter lab
```

To execute a notebook headless (e.g. to regenerate its outputs):

```bash
JAX_PLATFORMS=cpu uv run --group examples \
  jupyter nbconvert --to notebook --execute --inplace examples/01_standard_susie.ipynb
```

## The four axes

Every model composes four orthogonal choices — the notebooks vary one axis at a time:

- **design** — how genotypes map to the latent predictor: `SharedDesign` (one genotype
  matrix, multi-phenotype) or `BlockDesign` (stacked per-ancestry genotypes).
- **likelihood** — `gaussian`, `bernoulli_logit`, `poisson_log`, `cox`, `aft_weibull`,
  `mvn_resid`, … (or your own — see `docs/extending/likelihood.md`).
- **prior** — `susie` (univariate), `sushie` (one dense covariance), `mvsusie` (mixture
  of sharing patterns).
- **inclusion weights** — `log_pi` over variants (uniform by default).

> The notebooks are generated from [`_build_notebooks.py`](_build_notebooks.py) so the
> cell sources stay reviewable in version control. Edit that script and re-run it to
> regenerate the `.ipynb` files.
