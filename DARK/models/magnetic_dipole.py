from DARK import Model
import numpy as np
import DARK.constants as const


def get_model():

    c_dict = {
        "1": {
            "e": 0.25,  # factor of 1/4 in paper
            "p": -0.25,  # Factor of 1/4 in paper * (-1)
            "n": 0,
            "screened": True,
        },
        "4": {"e": 1, "p": -2.8, "n": 0},  # mu_tilde_e  # mu_tilde_p * (-1)
        "5a": {"e": 1, "p": -1, "n": 0, "screened": True},
        "5b": {"e": 1, "p": -1, "n": 0},
        "6": {"e": -1, "p": 2.8, "n": 0},  # -mu_tilde_e  # -mu_tilde_p * (-1)
    }

    def c_dict_form(op_id, particle_id, q_vec, mass, spin):
        """
        q/m_chi dependence of the c coefficients.

        Input:
            op_id : integer, operator id number
            particle_id : string, {"e", "p", "n"} for electron, proton, neutron resp.

            q_vec : (real, real, real), momentum vector in XYZ coordinates
            mass : dark matter mass
            spin : dark matter spin

        Output:
            real, the q/m_chi dependence of the c coefficients that isn't stored above in
            c_dict


        Note: To add different operators simply add more functions inside of here, and replace
            one_func in the output dict
        """

        def one_func(q, m_chi, spin):
            return 1.0

        def q_sq_on_mchi_sq(q, m_chi, spin):
            return np.dot(q, q) / m_chi**2

        def q_sq_on_mchi_me(q, m_chi, spin):
            return np.dot(q, q) / (m_chi * const.m_e)

        def q_sq_on_mchi_mp(q, m_chi, spin):
            return np.dot(q, q) / (m_chi * const.m_p)

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
        }[op_id][particle_id](q_vec, mass, spin)

    return Model("mdm", c_dict, c_dict_form)
