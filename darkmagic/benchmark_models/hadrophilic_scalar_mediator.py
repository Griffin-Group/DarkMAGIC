import numpy as np

from darkmagic import Model
from darkmagic.numerics import SphericalGrid

from darkmagic.benchmark_models.utils import one


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

        def F_med(grid: SphericalGrid) -> np.array:
            return np.ones_like(grid.q_norm)
    elif mass == "light":

        def F_med(grid: SphericalGrid) -> np.array:
            return grid.q_norm ** (-2)
    else:
        raise ValueError(
            "Unknown mass for the hadrophilic scalar mediator. "
            "Only 'light' and 'heavy' are supported."
        )

    return Model(
        f"{mass.capitalize()} Hadrophilic Scalar Mediator",
        coeff_prefactor,
        coeff_func,
        F_med,
    )


light_scalar_mediator = get_model("light")
heavy_scalar_mediator = get_model("heavy")
