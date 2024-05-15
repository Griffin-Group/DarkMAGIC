import numpy as np

import darkmagic.constants as const
from darkmagic.model import Model
from darkmagic.numerics import SphericalGrid

from darkmagic.benchmark_models.utils import one, reduced_mass


def get_model(mass: str) -> Model:
    """
    TODO: Add docstring.
    """
    coeff_prefactor = {
        "1": {"e": 0, "p": 1, "n": 1},
    }

    coeff_func = {
        "1": {"p": one, "n": one},
    }

    if mass == "heavy":

        def F_mediator_propagator(grid: SphericalGrid) -> np.array:
            return np.ones_like(grid.q_norm)

        def reference_cross_section(m_chi: np.array) -> np.array:
            return np.pi * reduced_mass(m_chi, const.m_n) ** (-2)
    elif mass == "light":

        def F_mediator_propagator(grid: SphericalGrid) -> np.array:
            return grid.q_norm ** (-2)

        def reference_cross_section(m_chi: np.array) -> np.array:
            q0 = m_chi * const.V0
            return np.pi * reduced_mass(m_chi, const.m_n) ** (-2) * (q0) ** 4

    else:
        raise ValueError(
            "Unknown mass for the hadrophilic scalar mediator. "
            "Only 'light' and 'heavy' are supported."
        )

    return Model(
        f"{mass.capitalize()} Hadrophilic Scalar Mediator",
        coeff_prefactor,
        coeff_func,
        F_mediator_propagator,
        reference_cross_section,
        shortname=f"{mass[0]}sm",
    )


light_scalar_mediator = get_model("light")
heavy_scalar_mediator = get_model("heavy")
