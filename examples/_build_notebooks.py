"""Generate the example notebooks from cell definitions using nbformat.

Run with:  uv run --group examples python examples/_build_notebooks.py
This keeps the .ipynb JSON always valid and the source under version control readable.
"""

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = __import__("pathlib").Path(__file__).parent


def build(path, cells):
    nb = new_notebook()
    nb.cells = [
        new_markdown_cell(src) if kind == "md" else new_code_cell(src)
        for kind, src in cells
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open(HERE / path, "w") as f:
        nbf.write(nb, f)
    print("wrote", path)


# --- shared preamble reused across notebooks --------------------------------------------
PREAMBLE = """\
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="notebook", style="whitegrid")
rng = np.random.default_rng(0)  # NumPy RNG for the simulations (deterministic)"""

# A small helper both PIP plots and CS tables use, defined per-notebook so each is
# self-contained.
PIP_HELPERS = '''\
def plot_pips(pip, causal, title, ax=None):
    """Stem plot of per-variant PIPs, with the true causal variants highlighted."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    m = np.arange(len(pip))
    ax.vlines(m, 0, pip, color="0.75", lw=1)
    ax.scatter(m, pip, s=12, color="0.4", label="variants")
    ax.scatter(causal, np.asarray(pip)[list(causal)], s=70, color="crimson",
               zorder=3, label="true causal")
    ax.set(xlabel="variant index", ylabel="PIP", ylim=(-0.02, 1.05), title=title)
    ax.legend(loc="upper right", frameon=True)
    return ax


def credible_set_table(result):
    """Tidy DataFrame summarising the credible sets that passed the purity filter."""
    rows = []
    for cs in result.credible_sets:
        if not cs.kept:
            continue
        rows.append({
            "effect": cs.effect,
            "size": len(cs.variants),
            "coverage": round(float(cs.coverage), 3),
            "purity": round(float(cs.purity), 3),
            "variants": list(np.asarray(cs.variants)[:8]),
        })
    return pd.DataFrame(rows)
'''


# =======================================================================================
# 1. Standard (univariate Gaussian) SuSiE
# =======================================================================================
build("01_standard_susie.ipynb", [
    ("md", """\
# 1 · Standard SuSiE (univariate Gaussian)

The classic [SuSiE](https://doi.org/10.1111/rssb.12388) fine-mapping setup: one
continuous phenotype, a region of correlated variants (LD), and a handful of true
causal signals hiding in the correlation. We want, for each variant, a **posterior
inclusion probability (PIP)**, and a small **credible set** per signal that is very
likely to contain the true causal variant.

In `lasusie` every model is built from four orthogonal axes:

| axis | this notebook |
|------|---------------|
| **design**     | `SharedDesign(X)` — one genotype matrix, `eta = X @ B` |
| **likelihood** | `gaussian(y, sigma2=...)` — identity link, EB residual variance |
| **prior**      | `susie(sigma0_sq=...)` — single Gaussian `N(0, sigma0^2)` |
| **inclusion**  | `log_pi` — uniform over variants |

`finemap` then runs the (variance-propagating) IBSS loop and returns PIPs + credible sets."""),

    ("code", PREAMBLE),
    ("code", PIP_HELPERS),

    ("md", """\
## Simulate a region with LD

We give the variants realistic linkage disequilibrium by sampling genotypes with an
AR(1) correlation `rho^|i-j|`, then standardise each column (SuSiE expects
mean-0 / unit-variance predictors). Two variants are truly causal."""),

    ("code", """\
N, M = 800, 150            # individuals, variants
rho = 0.9                  # AR(1) LD strength between neighbouring variants

idx = np.arange(M)
Sigma_ld = rho ** np.abs(idx[:, None] - idx[None, :])
chol = np.linalg.cholesky(Sigma_ld + 1e-6 * np.eye(M))
X = rng.standard_normal((N, M)) @ chol.T
X = (X - X.mean(0)) / X.std(0)

causal = [30, 100]         # true causal variant indices
betas = np.array([0.7, -0.5])
y = X[:, causal] @ betas + rng.standard_normal(N) * 1.0
print("X:", X.shape, " y:", y.shape, " causal:", causal)"""),

    ("md", """\
## Build the model and fine-map

`L` is the number of single effects (an upper bound on the number of signals). We ask
for 95% credible sets and drop any set whose purity (minimum absolute pairwise
correlation) falls below 0.5."""),

    ("code", """\
from lasusie import finemap, Model, likelihoods, priors
from lasusie.design import SharedDesign

model = Model(
    design=SharedDesign(X=jnp.asarray(X)),
    likelihood=likelihoods.gaussian(jnp.asarray(y).reshape(-1, 1), sigma2=1.0),
    prior=priors.susie(sigma0_sq=1.0),
    log_pi=jnp.full(M, -jnp.log(M)),
)

result = finemap(model, L=5, coverage=0.95, purity=0.5)
print(f"converged={result.converged}  iterations={result.iterations}  ELBO={result.elbo:.1f}")"""),

    ("md", "## Results: PIPs and credible sets"),

    ("code", """\
plot_pips(result.pip, causal, "Standard SuSiE — posterior inclusion probabilities")
plt.tight_layout(); plt.show()

credible_set_table(result)"""),

    ("md", """\
Each kept credible set should be a small cluster of variants in tight LD, and it should
contain one of the true causal variants (marked in red above). Because the two causal
variants sit in correlated neighbourhoods, the PIP mass for each signal is spread across
its LD block — exactly what a credible set is designed to capture.

### Where the numbers live

- `result.pip` — `(M,)` per-variant PIP
- `result.alpha` — `(L, M)` inclusion probabilities per single effect
- `result.credible_sets` — list of `CredibleSet(effect, variants, coverage, purity, kept)`
- `result.posterior.mean` — `(L, M, K)` posterior effect-size means (`K=1` here)
- `result.elbo_history` — ELBO per sweep (a convergence diagnostic)"""),

    ("code", """\
plt.figure(figsize=(6, 3))
plt.plot(result.elbo_history, marker="o", ms=3)
plt.xlabel("IBSS sweep"); plt.ylabel("ELBO"); plt.title("Convergence")
plt.tight_layout(); plt.show()"""),
])


# =======================================================================================
# 2. Survival SuSiE (Cox + parametric AFT)
# =======================================================================================
build("02_survival_susie.ipynb", [
    ("md", """\
# 2 · Survival fine-mapping (Cox & AFT)

Here the phenotype is a **survival time** with **right-censoring** — some individuals
have not had the event by the end of follow-up, so we only know their time exceeds a
threshold. `lasusie` ships two families of survival likelihoods:

- **`cox(times, events)`** — the Cox proportional-hazards *partial* likelihood. It is a
  *composite* likelihood: the risk-set denominators couple all individuals, so it does
  not factorise. No baseline hazard is estimated.
- **`aft_weibull` / `aft_lognormal` / `aft_loglogistic`** — *parametric*
  accelerated-failure-time models. These *do* factorise over individuals (censoring
  folded in per observation), so they get exact Gauss-Hermite variance propagation.

Both slot into the same `Model` — only the likelihood axis changes."""),

    ("code", PREAMBLE),
    ("code", PIP_HELPERS),

    ("md", """\
## Simulate survival times with a causal variant

We plant a single causal variant `j` driving the linear predictor `eta = X[:, j] * beta`.
Event times are exponential with hazard `exp(eta)` (a proportional-hazards truth), and we
apply independent random censoring so ~30% of individuals are censored."""),

    ("code", """\
N, M, j = 400, 60, 20
X = rng.standard_normal((N, M))
X = (X - X.mean(0)) / X.std(0)

beta = 1.3
eta_true = X[:, j] * beta

# Exponential event times with hazard exp(eta):  T = -log(U) / exp(eta)
U = rng.uniform(1e-6, 1.0, size=N)
event_time = -np.log(U) / np.exp(eta_true)

# Independent censoring times -> observe min(event, censor)
censor_time = rng.exponential(scale=np.quantile(event_time, 0.8), size=N)
times = np.minimum(event_time, censor_time)
events = (event_time <= censor_time).astype(float)   # 1 = observed, 0 = censored
print(f"observed events: {events.mean():.0%}   causal variant: {j}")"""),

    ("md", """\
## Cox proportional hazards

The Cox likelihood needs only the times and the event indicator. Everything else — prior,
design, inclusion weights — is identical to the Gaussian case.

> Because Cox is a *composite* likelihood, its offset-variance propagation goes through a
> dense delta-method correction (an `N x N` Hessian). For a quick fit we pass
> `propagate_variance=False`, the cheaper zeroth-order option — perfectly fine for a demo
> and much faster on CPU. Drop it (the default `True`) for the full variance-propagating
> fit."""),

    ("code", """\
from lasusie import finemap, Model, likelihoods, priors
from lasusie.design import SharedDesign

Xj = jnp.asarray(X)
cox_model = Model(
    design=SharedDesign(X=Xj),
    likelihood=likelihoods.cox(jnp.asarray(times), jnp.asarray(events)),
    prior=priors.susie(sigma0_sq=1.0),
    log_pi=jnp.full(M, -jnp.log(M)),
)
cox_res = finemap(cox_model, L=3, coverage=0.95, purity=0.5, propagate_variance=False)
print(f"Cox: top variant = {int(np.argmax(cox_res.pip))}  (true = {j}),  "
      f"PIP = {cox_res.pip[j]:.3f}")"""),

    ("md", """\
## Parametric AFT (Weibull)

A Weibull AFT models `log T = eta + scale * W` with an extreme-value error. Same data,
different likelihood — note it takes the same `(times, events)` plus a fixed `scale`."""),

    ("code", """\
aft_model = Model(
    design=SharedDesign(X=Xj),
    likelihood=likelihoods.aft_weibull(jnp.asarray(times), jnp.asarray(events), scale=1.0),
    prior=priors.susie(sigma0_sq=1.0),
    log_pi=jnp.full(M, -jnp.log(M)),
)
aft_res = finemap(aft_model, L=3, coverage=0.95, purity=0.5)
print(f"AFT-Weibull: top variant = {int(np.argmax(aft_res.pip))}  (true = {j}),  "
      f"PIP = {aft_res.pip[j]:.3f}")"""),

    ("md", "## Compare the two survival models"),

    ("code", """\
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
plot_pips(cox_res.pip, [j], "Cox proportional hazards", ax=axes[0])
plot_pips(aft_res.pip, [j], "Weibull AFT", ax=axes[1])
plt.tight_layout(); plt.show()

print("Cox credible sets:");        display(credible_set_table(cox_res))
print("AFT-Weibull credible sets:"); display(credible_set_table(aft_res))"""),

    ("md", """\
Both survival models concentrate the PIP on the true causal variant despite ~30%
censoring. Cox makes no assumption about the baseline hazard shape; the parametric AFT
is cheaper (it factorises) and additionally gives you an interpretable time-scale via
`scale`. Swap `aft_weibull` for `aft_lognormal` or `aft_loglogistic` to change the
baseline error distribution."""),
])


# =======================================================================================
# 3. Multi-phenotype SuSiE (Gaussian, non-Gaussian, mixed)
# =======================================================================================
build("03_multiphenotype_susie.ipynb", [
    ("md", """\
# 3 · Multi-phenotype fine-mapping

Now every individual is measured for **K phenotypes** and we want to borrow strength
across them: a variant causal for several correlated traits should be easier to find
jointly than one trait at a time.

The design is `SharedDesign(X)` — one genotype matrix, `eta = X @ B` of shape `(N, K)`.
Phenotypes are coupled through **two** channels:

1. the **prior** on the K-vector effect (`sushie` = one dense covariance, or `mvsusie`
   = a *mixture* of sharing patterns), and
2. the **likelihood**, when it correlates the phenotype residuals (`mvn_resid`).

We show three flavours:

- **Gaussian** multi-phenotype via `mvn_resid` (residual covariance across traits),
- **non-Gaussian** multi-phenotype (two binary traits, `bernoulli_logit`),
- a **mixed** likelihood (one continuous + one binary trait) via a tiny custom composite."""),

    ("code", PREAMBLE),
    ("code", PIP_HELPERS),

    ("md", """\
## A · Gaussian multi-phenotype (`mvn_resid`)

Two correlated continuous phenotypes share a causal variant `j`. `mvn_resid` carries the
`K x K` residual covariance `Sigma_e`; the `sushie` prior puts a single dense covariance
on the effect vector so the two traits' effects are estimated jointly."""),

    ("code", """\
from lasusie import finemap, Model, likelihoods, priors
from lasusie.design import SharedDesign

N, M, j = 400, 80, 25
K = 2
X = rng.standard_normal((N, M))
X = (X - X.mean(0)) / X.std(0)

effects = np.array([1.2, 0.9])                 # per-phenotype effect of variant j
resid_rho = 0.5                                # residual correlation across phenotypes
Sigma_e = np.array([[1.0, resid_rho], [resid_rho, 1.0]])
noise = rng.standard_normal((N, K)) @ np.linalg.cholesky(Sigma_e).T
Y = np.outer(X[:, j], effects) + noise
print("Y:", Y.shape)"""),

    ("code", """\
gauss_model = Model(
    design=SharedDesign(X=jnp.asarray(X)),
    likelihood=likelihoods.mvn_resid(jnp.asarray(Y), resid_cov=jnp.asarray(Sigma_e)),
    prior=priors.sushie(jnp.eye(K)),
    log_pi=jnp.full(M, -jnp.log(M)),
)
gauss_res = finemap(gauss_model, L=3, coverage=0.95, purity=0.5)
print(f"top variant = {int(np.argmax(gauss_res.pip))}  (true = {j}),  PIP = {gauss_res.pip[j]:.3f}")

plot_pips(gauss_res.pip, [j], "Gaussian multi-phenotype (mvn_resid + sushie)")
plt.tight_layout(); plt.show()
credible_set_table(gauss_res)"""),

    ("md", """\
### The mixture prior (mvSuSiE)

If you don't know *how* an effect is shared across traits, use `mvsusie` with
`canonical_components`: a mixture over standard sharing patterns (fully shared,
independent, trait-specific, null). The mixing weights are learned by empirical Bayes.

> **Which `ser_fit`?** For a **Gaussian** likelihood the conjugate SER is *exact* and needs
> only one Laplace fit per variant, so we pass `ser_fit="conjugate"` here — both exact and
> cheaper than the default `"joint"` (which fits the MAP of each mixture component
> separately, `G` fits per variant); the two agree to floating-point precision on a Gaussian
> likelihood. `"joint"` earns its extra cost on **non-Gaussian** mixtures, where the
> conjugate path's likelihood-only mode can diverge (a count / exp-family null variant has no
> interior maximum) and the per-component prior curvature `Σ_g⁻¹` regularises it at the
> source. Both paths handle the rank-deficient canonical components (e.g. the fully-shared
> `[[1,1],[1,1]]`) in single or double precision — the joint fit keeps them invertible with a
> dtype-relative ridge."""),

    ("code", """\
from lasusie.priors import canonical_components

mix_model = Model(
    design=SharedDesign(X=jnp.asarray(X)),
    likelihood=likelihoods.mvn_resid(jnp.asarray(Y)),          # default Sigma_e = I, EB-updated
    prior=priors.mvsusie(canonical_components(K=K, scale=1.0)),
    log_pi=jnp.full(M, -jnp.log(M)),
)
mix_res = finemap(mix_model, L=3, coverage=0.95, purity=0.5,
                  update_prior=True, ser_fit="conjugate")
print(f"mvSuSiE mixture: top variant = {int(np.argmax(mix_res.pip))}  (true = {j}),  "
      f"PIP = {mix_res.pip[j]:.3f}")"""),

    ("md", """\
## B · Non-Gaussian multi-phenotype (two binary traits)

`SharedDesign` gives `eta` of shape `(N, K)`; a pointwise likelihood like
`bernoulli_logit` is just applied elementwise, so it handles `K` binary phenotypes with
no extra machinery. The traits are coupled through the `sushie` prior on the shared
effect vector."""),

    ("code", """\
lin = np.outer(X[:, j], np.array([1.5, 1.2]))          # logit-scale signal at variant j
probs = 1.0 / (1.0 + np.exp(-lin))
Yb = rng.binomial(1, probs).astype(float)              # (N, K) binary phenotypes
print("Yb balance per trait:", Yb.mean(0).round(2))

bern_model = Model(
    design=SharedDesign(X=jnp.asarray(X)),
    likelihood=likelihoods.bernoulli_logit(jnp.asarray(Yb)),
    prior=priors.sushie(jnp.eye(K) * 4.0),             # wider prior for logit-scale effects
    log_pi=jnp.full(M, -jnp.log(M)),
)
bern_res = finemap(bern_model, L=3, coverage=0.95, purity=0.5)
print(f"binary multi-pheno: top variant = {int(np.argmax(bern_res.pip))}  (true = {j}),  "
      f"PIP = {bern_res.pip[j]:.3f}")

plot_pips(bern_res.pip, [j], "Two binary phenotypes (bernoulli_logit + sushie)")
plt.tight_layout(); plt.show()"""),

    ("md", """\
## C · Mixed likelihood (continuous + binary)

What if the phenotypes are of *different types* — say one continuous and one binary? A
likelihood only has to answer three questions (`log_density`, `expected_log_density`,
`updated`; see the [extending guide](../docs/extending/likelihood.md)). We can build a
**mixed** likelihood by delegating each column of `eta` to a per-trait pointwise
likelihood and summing. The built-in `gaussian` / `bernoulli_logit` factories already
provide exact per-column expectations, so the composite is a few lines."""),

    ("code", '''\
import equinox as eqx


class MixedColumns(eqx.Module):
    """A per-column mix of pointwise likelihoods for multi-phenotype `eta` of shape (N, K).

    Column k of `eta` is scored by `liks[k]`. Because each built-in pointwise likelihood
    already supplies an (exact, Gauss-Hermite) `expected_log_density` and an EB `updated`,
    the composite just delegates column-by-column and sums.
    """

    liks: list

    def log_density(self, eta):
        return sum(lik.log_density(eta[:, k]) for k, lik in enumerate(self.liks))

    def expected_log_density(self, mean, var):
        return sum(lik.expected_log_density(mean[:, k], var[:, k])
                   for k, lik in enumerate(self.liks))

    def updated(self, eta_mean, eta_var):
        return MixedColumns(liks=[lik.updated(eta_mean[:, k], eta_var[:, k])
                                  for k, lik in enumerate(self.liks)])
'''),

    ("code", """\
# phenotype 0: continuous Gaussian;  phenotype 1: binary
y_cont = X[:, j] * 1.0 + rng.standard_normal(N) * 1.0
p_bin = 1.0 / (1.0 + np.exp(-(X[:, j] * 1.5)))
y_bin = rng.binomial(1, p_bin).astype(float)

mixed_lik = MixedColumns(liks=[
    likelihoods.gaussian(jnp.asarray(y_cont), sigma2=1.0),   # column 0
    likelihoods.bernoulli_logit(jnp.asarray(y_bin)),         # column 1
])

mixed_model = Model(
    design=SharedDesign(X=jnp.asarray(X)),
    likelihood=mixed_lik,
    prior=priors.sushie(jnp.eye(K) * 2.0),
    log_pi=jnp.full(M, -jnp.log(M)),
)
mixed_res = finemap(mixed_model, L=3, coverage=0.95, purity=0.5)
print(f"mixed likelihood: top variant = {int(np.argmax(mixed_res.pip))}  (true = {j}),  "
      f"PIP = {mixed_res.pip[j]:.3f}")

plot_pips(mixed_res.pip, [j], "Mixed likelihood (Gaussian + Bernoulli)")
plt.tight_layout(); plt.show()
credible_set_table(mixed_res)"""),

    ("md", """\
The same `Model` machinery fine-maps a continuous and a binary phenotype jointly, sharing
information through the prior on the effect vector. The custom `MixedColumns` class is all
it took — any object with `log_density` / `expected_log_density` / `updated` is a valid
likelihood axis."""),
])


# =======================================================================================
# 4. Multi-ancestry SuSiE
# =======================================================================================
build("04_multiancestry_susie.ipynb", [
    ("md", """\
# 4 · Multi-ancestry fine-mapping

In a multi-ancestry study the individuals are **disjoint across ancestries** — each
ancestry has its own genotype matrix and its own samples — but we believe they share
(largely) the same causal variants, possibly with ancestry-specific effect sizes.
Fine-mapping jointly exploits the *different LD patterns* across ancestries to narrow the
credible set far more than any single ancestry could.

This is the `BlockDesign` axis: genotypes are stacked as `(K, N, M)` and
`eta[k] = X[k] @ B[:, k]` has shape `(K, N)`. The `sushie` prior places a dense `K x K`
covariance on the per-ancestry effect vector, letting effects correlate across ancestries
while still allowing them to differ."""),

    ("code", PREAMBLE),

    ("md", """\
## Simulate K ancestries with distinct LD

Each ancestry gets its own genotype matrix (here with its own AR(1) LD strength, so the
correlation structure genuinely differs across ancestries) and its own samples. One
shared causal variant `j` has ancestry-specific effect sizes."""),

    ("code", """\
K, N, M, j = 3, 350, 150, 70
ld_strengths = [0.9, 0.7, 0.5]              # different LD decay per ancestry
effects = np.array([1.2, 0.9, 1.4])         # ancestry-specific effect of variant j

idx = np.arange(M)
X = np.empty((K, N, M))
y = np.empty((K, N))
for k in range(K):
    Sigma_ld = ld_strengths[k] ** np.abs(idx[:, None] - idx[None, :])
    chol = np.linalg.cholesky(Sigma_ld + 1e-6 * np.eye(M))
    Xk = rng.standard_normal((N, M)) @ chol.T
    Xk = (Xk - Xk.mean(0)) / Xk.std(0)      # standardise within ancestry
    X[k] = Xk
    y[k] = Xk[:, j] * effects[k] + rng.standard_normal(N)
print("X:", X.shape, " y:", y.shape, " causal:", j)"""),

    ("md", """\
## Fine-map with `BlockDesign` + `sushie`

The Gaussian likelihood is applied to the stacked `(K, N)` responses; the `BlockDesign`
routes each ancestry's effects through its own genotypes.

`L` is an *upper bound* on the number of signals — we set `L=3` even though there is one
true causal. Unused single effects have their prior variance shrunk toward zero by
empirical Bayes and spread their (tiny) inclusion mass diffusely, so they don't survive
the credible-set purity filter: we get back exactly one credible set."""),

    ("code", """\
from lasusie import finemap, Model, likelihoods, priors
from lasusie.design import BlockDesign

model = Model(
    design=BlockDesign(X=jnp.asarray(X)),
    likelihood=likelihoods.gaussian(jnp.asarray(y), sigma2=1.0),
    prior=priors.sushie(jnp.eye(K)),
    log_pi=jnp.full(M, -jnp.log(M)),
)
result = finemap(model, L=3, coverage=0.95, purity=0.5)
print(f"top variant = {int(np.argmax(result.pip))}  (true = {j}),  PIP = {result.pip[j]:.3f}")"""),

    ("md", "## PIPs and credible sets"),

    ("code", """\
def credible_set_table(result):
    rows = []
    for cs in result.credible_sets:
        if not cs.kept:
            continue
        rows.append({"effect": cs.effect, "size": len(cs.variants),
                     "coverage": round(float(cs.coverage), 3),
                     "purity": round(float(cs.purity), 3),
                     "variants": list(np.asarray(cs.variants)[:8])})
    return pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(9, 3))
m = np.arange(M)
ax.vlines(m, 0, result.pip, color="0.75", lw=1)
ax.scatter(m, result.pip, s=12, color="0.4", label="variants")
ax.scatter(j, result.pip[j], s=70, color="crimson", zorder=3, label="true causal")
ax.set(xlabel="variant index", ylabel="PIP", ylim=(-0.02, 1.05),
       title="Multi-ancestry SuSiE (BlockDesign + sushie)")
ax.legend(); plt.tight_layout(); plt.show()

credible_set_table(result)"""),

    ("md", """\
## Recovered ancestry-specific effect sizes

Because the prior is a dense `K x K` covariance, each single effect stores a *vector* of
per-ancestry effect sizes in `result.posterior.mean` (shape `(L, M, K)`). Reading it at
the causal variant recovers the ancestry-specific effects we simulated."""),

    ("code", """\
# which single effect (l) captured the causal variant?
alpha = np.asarray(result.alpha)          # (L, M)
l_star = int(np.argmax(alpha[:, j]))
est = np.asarray(result.posterior.mean)[l_star, j]    # (K,) per-ancestry effect estimate

comparison = pd.DataFrame({
    "ancestry": [f"anc{k}" for k in range(K)],
    "true_effect": effects,
    "estimated_effect": est.round(3),
})
comparison"""),

    ("code", """\
fig, ax = plt.subplots(figsize=(6, 4))
xpos = np.arange(K)
ax.bar(xpos - 0.2, effects, width=0.4, label="true", color="0.6")
ax.bar(xpos + 0.2, est, width=0.4, label="estimated", color="crimson", alpha=0.8)
ax.set_xticks(xpos); ax.set_xticklabels([f"ancestry {k}" for k in range(K)])
ax.set_ylabel("effect size"); ax.set_title(f"Per-ancestry effects at causal variant {j}")
ax.legend(); plt.tight_layout(); plt.show()"""),

    ("md", """\
Jointly modelling the three ancestries — each with a *different* LD pattern — pins the
signal to the true causal variant and recovers its ancestry-specific effect sizes, all
from a single `finemap` call with the `BlockDesign` axis."""),
])

print("done")
