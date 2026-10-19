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
from .genetic import BREEDERS, breed, evolve
from .models.fft import FFT
from .operators.coherence import PartialCoherence, UpdateCoherence, partial_coherence
from .operators.modulus import ModulusProjection
from .operators.phase import PhaseLimited, phase_limit
from .operators.shrinkwrap import Shrinkwrap
from .operators.support import SupportProjection
from .postprocess import align_to, register, remove_ramp
from .prtf import PRTF, prtf, prtf_curve
from .recipes.recipe import Recipe, detwin, repeat

__all__ = [
    "BREEDERS",
    "CDI",
    "Diagnosis",
    "ER",
    "FFT",
    "HIO",
    "RAAR",
    "Ensemble",
    "ModulusProjection",
    "PRTF",
    "PartialCoherence",
    "PhaseLimited",
    "ReconState",
    "Reconstruction",
    "Recipe",
    "Shrinkwrap",
    "SupportProjection",
    "UpdateCoherence",
    "align_to",
    "breed",
    "default_recipe",
    "detwin",
    "diagnose",
    "evolve",
    "prtf",
    "prtf_curve",
    "partial_coherence",
    "phase_limit",
    "reconstruct",
    "register",
    "remove_ramp",
    "repeat",
]
