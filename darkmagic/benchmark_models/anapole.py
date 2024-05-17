import numpy as np

import darkmagic.constants as const
from darkmagic.model import Model
from darkmagic.benchmark_models.utils import q_sq_on_mchi_sq, reduced_mass


def get_model() -> Model:
    """
    TODO: Add docstring.
    """
    coeff_prefactor = {
        "8a": {
            "e": 1 / 2,  # factor of 1/2 in paper
            "p": -1 / 2,  # Factor of 1/2 in paper * (-1)
        },
        "8b": {
            "e": 1 / 2,  # factor of 1/2 in paper
            "p": -1 / 2,  # Factor of 1/2 in paper * (-1)
        },
        "9": {
            "e": -const.mu_tilde_e / 2,  # -mu_tilde_e/2
            "p": const.mu_tilde_p / 2,  # -mu_tilde_p/2 * (-1)
        },
    }

    coeff_func = {
        "8a": {"e": q_sq_on_mchi_sq, "p": q_sq_on_mchi_sq},
        "8b": {"e": q_sq_on_mchi_sq, "p": q_sq_on_mchi_sq},
        "9": {"e": q_sq_on_mchi_sq, "p": q_sq_on_mchi_sq},
    }

    def reference_cross_section(m_chi: float) -> float:
        return (
            np.pi
            / 6
            * 1
            / (const.alphaEM**2 * const.m_e**2 * reduced_mass(m_chi, const.m_e) ** 2)
        )

    return Model("Anapole", coeff_prefactor, coeff_func)


anapole = get_model()
