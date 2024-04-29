from numpy import linalg as LA
from pytest_parametrize_cases import Case, parametrize_cases

import darkmagic.constants as const
from darkmagic import Model


def get_model_1():
    coeff = {
        "1": {"e": 1 / 4, "p": -1 / 4, "n": 1},
        "2": {"e": 1 / 4, "p": -1 / 4, "n": 1},
        "3": {"e": 1 / 4, "p": -1 / 4, "n": 1},
        "4": {"e": const.mu_tilde_e, "p": -const.mu_tilde_p, "n": 1},
        "5a": {"e": 1, "p": -1, "n": 1},
        "5b": {"e": 1, "p": -1, "n": 1},
        "6": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "7a": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "7b": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "8a": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "8b": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "9": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "10": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "11": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "12a": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "12b": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "13a": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "13b": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "14a": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "14b": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "15a": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
        "15b": {"e": -const.mu_tilde_e, "p": const.mu_tilde_p, "n": 1},
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
            "2": {"e": q_sq_on_mchi_sq, "p": q_sq_on_mchi_sq, "n": one_func},
            "3": {"e": q_sq_on_mchi_sq, "p": q_sq_on_mchi_sq, "n": one_func},
            "4": {"e": q_sq_on_mchi_me, "p": q_sq_on_mchi_mp, "n": one_func},
            "5a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "5b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "6": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "7a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "7b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "8a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "8b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "9": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "10": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "11": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "12a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "12b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "13a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "13b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "14a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "14b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "15a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "15b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
        }[op_id][particle_id](q, m_chi, S_chi)

    return Model("test", coeff, coeff_qmS)


def get_model_2():
    coeff = {
        "2": {"p": -1 / 4, "n": 1},
        "3": {"p": -1 / 4, "n": 1},
        "4": {"p": -const.mu_tilde_p, "n": 1},
        "5a": {"p": -1, "n": 1},
        "5b": {"p": -1, "n": 1},
        "6": {"p": const.mu_tilde_p, "n": 1},
        "7a": {"p": const.mu_tilde_p, "n": 1},
        "7b": {"p": const.mu_tilde_p, "n": 1},
        "8a": {"p": const.mu_tilde_p, "n": 1},
        "8b": {"e": 0, "p": const.mu_tilde_p, "n": 1},
        "9": {"e": 0, "p": const.mu_tilde_p, "n": 1},
        "10": {"e": 0, "p": const.mu_tilde_p, "n": 1},
        "11": {"e": 0, "p": const.mu_tilde_p, "n": 1},
        "12a": {"e": 0, "p": const.mu_tilde_p, "n": 1},
        "12b": {"e": 0, "p": const.mu_tilde_p, "n": 1},
        "13a": {"e": 0, "p": const.mu_tilde_p, "n": 1},
        "13b": {"e": 0, "p": const.mu_tilde_p, "n": 1},
        "14b": {"e": 0, "p": const.mu_tilde_p, "n": 1},
        "15b": {"e": 0, "p": const.mu_tilde_p, "n": 1},
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
            "2": {"e": q_sq_on_mchi_sq, "p": q_sq_on_mchi_sq, "n": one_func},
            "3": {"e": q_sq_on_mchi_sq, "p": q_sq_on_mchi_sq, "n": one_func},
            "4": {"e": q_sq_on_mchi_me, "p": q_sq_on_mchi_mp, "n": one_func},
            "5a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "5b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "6": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "7a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "7b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "8a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "8b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "9": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "10": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "11": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "12a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "12b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "13a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "13b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "14a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "14b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "15a": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
            "15b": {"e": me_on_mchi, "p": mp_on_mchi, "n": one_func},
        }[op_id][particle_id](q, m_chi, S_chi)

    return Model("test", coeff, coeff_qmS)


ALL_OPERATORS = {
    "1",
    "2",
    "3",
    "4",
    "5a",
    "5b",
    "6",
    "7a",
    "7b",
    "8a",
    "8b",
    "9",
    "10",
    "11",
    "12a",
    "12b",
    "13a",
    "13b",
    "14a",
    "14b",
    "15a",
    "15b",
}


@parametrize_cases(
    Case(
        "omni_model",
        model=get_model_1(),
        particles={"e", "p", "n"},
        operators=ALL_OPERATORS,
    ),
    Case(
        "some_missing",
        model=get_model_2(),
        particles={"n", "p"},
        operators=ALL_OPERATORS - {"1", "14a", "15a"},
    ),
)
def test_particles_operators(model, particles, operators):
    assert model.particles == particles
    assert model.operators == operators
