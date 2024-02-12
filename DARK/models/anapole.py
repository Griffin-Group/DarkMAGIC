from DARK.core import Model
import numpy as np
import DARK.constants as const


def get_model():
    c_dict = {
        "8a": {
            "e": 1 / 2,  # factor of 1/2 in paper
            "p": -1 / 2,  # Factor of 1/2 in paper * (-1)
            "n": 0,
            "screened": True,
        },
        "8b": {
            "e": 1 / 2,  # factor of 1/2 in paper
            "p": -1 / 2,  # Factor of 1/2 in paper * (-1)
            "n": 0,
        },
        "9": {
            "e": -1 / 2,  # -mu_tilde_e/2
            "p": 2.8 / 2,  # -mu_tilde_p/2 * (-1)
            "n": 0,
            "screened": True,
        },
    }

    def c_dict_form(op_id, particle_id, q_vec, mass, spin):
        def one_func(q_vec, mass, spin):
            return 1.0

        def q_sq_on_mchi_sq(q_vec, mass, spin):
            return np.dot(q_vec, q_vec) / mass**2

        def q_sq_on_mchi_me(q_vec, mass, spin):
            return np.dot(q_vec, q_vec) / (mass * const.m_e)

        def q_sq_on_mchi_mp(q_vec, mass, spin):
            return np.dot(q_vec, q_vec) / (mass * const.m_p)

        def me_on_mchi(q_vec, mass, spin):
            return const.m_e / mass

        def mp_on_mchi(q_vec, mass, spin):
            return const.m_p / mass

        return {
            "8a": {"e": q_sq_on_mchi_sq, "p": q_sq_on_mchi_sq, "n": one_func},
            "8b": {"e": q_sq_on_mchi_sq, "p": q_sq_on_mchi_sq, "n": one_func},
            "9": {"e": q_sq_on_mchi_sq, "p": q_sq_on_mchi_sq, "n": one_func},
        }[op_id][particle_id](q_vec, mass, spin)

    return Model("ap", c_dict, c_dict_form)
