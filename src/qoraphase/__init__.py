"""QoraPhase -- a JAX-native, composable phase-retrieval framework for CDI.

Array in, reconstruction out::

    import qoraphase as qp
    result = qp.reconstruct(intensity, support=support)

All public arrays are center-origin, ``mask=True`` means an unusable pixel, and
``intensity`` is a measured intensity (not an amplitude).  See spec section 4.
"""

from .algorithms.er import ER
from .algorithms.hio import HIO
from .algorithms.raar import RAAR
from .api import default_recipe, reconstruct
from .core.ensemble import Ensemble
from .core.problem import CDI
from .core.result import Reconstruction
from .core.state import ReconState
from .diagnostics import Diagnosis, diagnose
from .models.fft import FFT
from .operators.modulus import ModulusProjection
from .operators.shrinkwrap import Shrinkwrap
from .operators.support import SupportProjection
from .postprocess import align_to
from .recipes.recipe import Recipe, detwin, repeat

__all__ = [
    "CDI",
    "Diagnosis",
    "ER",
    "FFT",
    "HIO",
    "RAAR",
    "Ensemble",
    "ModulusProjection",
    "ReconState",
    "Reconstruction",
    "Recipe",
    "Shrinkwrap",
    "SupportProjection",
    "align_to",
    "default_recipe",
    "detwin",
    "diagnose",
    "reconstruct",
    "repeat",
]
