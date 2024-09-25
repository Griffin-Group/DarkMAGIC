import numpy as np

import darkmagic.constants as const
from darkmagic.benchmark_models.utils import one, reduced_mass
from darkmagic.model import Model
from darkmagic.numerics import SphericalGrid


def get_model() -> Model:
    """
    TODO: Add docstring.
    """
    coeff_prefactor = {
        "1": {"e": 1, "p": -1, "n": 0},
    }

    coeff_func = {
        "1": {"e": one, "p": one},
    }

    def F_mediator_propagator(grid: SphericalGrid) -> np.array:
        return grid.q_norm ** (-2)

    def reference_cross_section(m_chi: np.array) -> np.array:
        q0 = const.alpha_FS * const.m_e
        return np.pi * reduced_mass(m_chi, const.m_e) ** (-2) * (q0) ** 4

    return Model(
        "Dark Photon",
        coeff_prefactor,
        coeff_func,
        F_mediator_propagator,
        reference_cross_section,
        shortname="dp",
    )


dark_photon = get_model()
