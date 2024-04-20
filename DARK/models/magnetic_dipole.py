import numpy as np

from DARK import Model
import DARK.constants as const

from numpy import linalg as LA


def get_model():

    coeff = {
        "1": {"e": 1 / 4, "p": -1 / 4, "n": 0},
        "4": {"e": const.mu_tilde_e, "p": -const.mu_tilde_p, "n": 0},
        "5a": {"e": 1, "p": -1, "n": 0},
        "5b": {"e": 1, "p": -1, "n": 0},
        "6": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 0},
    }

    def coeff_qmS(op_id, particle_id, q, m_chi, S_chi):
        def one_func(q, m_chi, spin):
            return 1.0

        def q_sq_on_mchi_sq(q, m_chi, spin):
            return LA.norm(q, axis=1) / m_chi**2

        def q_sq_on_mchi_me(q, m_chi, spin):
            return LA.norm(q, axis=1) / (m_chi * const.m_e)

        def q_sq_on_mchi_mp(q, m_chi, spin):
            return LA.norm(q, axis=1) / (m_chi * const.m_p)

        def me_on_mchi(q, m_chi, spin):
            return const.m_e / m_chi

        def mp_on_mchi(q, m_chi, spin):
            return const.m_p / m_chi

        return {
            "1": {"e": q_sq_on_mchi_sq, "p": q_sq_on_mchi_sq, "n": one_func},
            "4": {"e": q_sq_on_mchi_me, "p": q_sq_on_mchi_mp, "n": one_func},
            "5a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "5b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "6": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
        }[op_id][particle_id](q, m_chi, S_chi)

    return Model("mdm", coeff, coeff_qmS)
