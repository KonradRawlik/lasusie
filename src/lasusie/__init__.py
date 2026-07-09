"""lasusie: generalized SuSiE fine-mapping in JAX.

Three orthogonal axes compose into a :class:`~lasusie.model.Model`:
  * a :mod:`~lasusie.design` map (genotype -> latent predictor geometry),
  * a flexible :mod:`~lasusie.likelihoods` (pointwise or composite),
  * a mixture-of-Gaussians effect :mod:`~lasusie.priors`.

``finemap`` runs the variance-propagating IBSS loop and returns PIPs + credible sets.
"""

from . import likelihoods, priors
from .api import CredibleSet, FineMapResult, finemap
from .covariates import (
    BlockCovariates,
    Covariates,
    SharedCovariates,
    block_covariates,
    shared_covariates,
)
from .design import BlockDesign, DesignMap, SharedDesign
from .ibss import SuSiEResult, ibss
from .laplace import GaussianPotential
from .model import (
    AbstractLikelihood,
    Likelihood,
    Model,
    PointwiseLikelihood,
    Prior,
    likelihood,
)

__all__ = [
    "finemap",
    "FineMapResult",
    "CredibleSet",
    "ibss",
    "SuSiEResult",
    "Model",
    "Prior",
    "AbstractLikelihood",
    "PointwiseLikelihood",
    "Likelihood",
    "likelihood",
    "GaussianPotential",
    "DesignMap",
    "SharedDesign",
    "BlockDesign",
    "Covariates",
    "SharedCovariates",
    "BlockCovariates",
    "shared_covariates",
    "block_covariates",
    "priors",
    "likelihoods",
]
