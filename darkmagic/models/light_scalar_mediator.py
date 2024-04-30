import numpy as np

from darkmagic import Model
from darkmagic.numerics import SphericalGrid


def get_model():
    coeff_prefactor = {
        "1": {"e": 0, "p": 1, "n": 1},
    }

    def one_func(q, m_chi, spin):
        return 1.0

    coeff_func = {
        "1": {"e": one_func, "p": one_func, "n": one_func},
    }

    def F_med(grid: SphericalGrid) -> np.array:
        return grid.q_norm ** (-2)

    return Model("lsm", coeff_prefactor, coeff_func, F_med)
