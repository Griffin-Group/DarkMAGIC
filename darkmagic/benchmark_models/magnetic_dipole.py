import numpy as np

import darkmagic.constants as const
from darkmagic import Model
from darkmagic.benchmark_models.utils import (
    me_on_mchi,
    mp_on_mchi,
    q_sq_on_mchi_me,
    q_sq_on_mchi_mp,
    q_sq_on_mchi_sq,
)


def get_model() -> Model:
    """
    TODO: Add docstring.
    """
    coeff_prefactor = {
        "1": {"e": 1 / 4, "p": -1 / 4, "n": 0},
        "4": {"e": const.mu_tilde_e, "p": -const.mu_tilde_p, "n": 0},
        "5a": {"e": 1, "p": -1, "n": 0},
        "5b": {"e": 1, "p": -1, "n": 0},
        "6": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 0},
    }

    coeff_func = {
        "1": {"e": q_sq_on_mchi_sq, "p": q_sq_on_mchi_sq},
        "4": {"e": q_sq_on_mchi_me, "p": q_sq_on_mchi_mp},
        "5a": {"e": me_on_mchi, "p": mp_on_mchi},
        "5b": {"e": me_on_mchi, "p": mp_on_mchi},
        "6": {"e": me_on_mchi, "p": mp_on_mchi},
    }

    def reference_cross_section(m_chi: float) -> float:
        return (
            np.pi
            * (m_chi + const.m_e) ** 2
            / (6 * m_chi**2 + const.m_e**2)
            / const.m_e**2
        )

    return Model(
        "Magnetic Dipole",
        coeff_prefactor,
        coeff_func,
        ref_cross_sect=reference_cross_section,
        shortname="mdm",
    )


magnetic_dipole = get_model()
