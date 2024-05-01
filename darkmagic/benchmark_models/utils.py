from numpy.typing import ArrayLike
import numpy as np
import darkmagic.constants as const


def reduced_mass(m1: ArrayLike, m2: float):
    m1 = np.array(m1)
    return m1 * m2 / (m1 + m2)


# TODO: these need reworking to work with q-grids
def one(q_vec, mass, spin):
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
