import numpy as np
import DARK.constants as const


class MBIntegrals:
    """
    Class for truncated Maxwell-Boltzmann integrals
    """
    def __init__(self) -> None:
        pass

def matrix_vminus(q, omega, m_chi, v_e):
    """
    Computes v_minus = min(|v_star|, Vesc) for each (q, omega) pair
    q: numpy array of shape (n_q, 3), momentum transfer
    omega: 2D numpy array of shape (n_q, n_modes), eigenmodes
    m_chi: float, DM mass
    v_e: vector, lab frame velocity of earth
    """
    n_modes = omega.shape[1]
    tiled_q = np.tile(q[None, :, :], (n_modes, 1, 1)).swapaxes(0, 1)
    q_norm = np.linalg.norm(tiled_q, axis=2)
    v_star = 1 / q_norm * (np.dot(tiled_q, v_e) + q_norm**2 / 2 / m_chi + omega)
    return np.minimum(np.abs(v_star), const.VESC)


def matrix_g0(q, omega, m_chi, v_e):
    """
    Computes the g0 integral for each (q, omega) pair
    See Eq. C9 in EFT paper (2009.13534)
    """

    v_minus = matrix_vminus(q, omega, m_chi, v_e)
    c1 = 2 * np.pi**2 * const.V0**2 / (np.linalg.norm(q, axis=1) * const.N0)
    c1 = np.tile(c1, (omega.shape[1], 1)).T

    return c1 * (
        np.exp(-(v_minus**2) / const.V0**2) - np.exp(-const.VESC**2 / const.V0**2)
    )


def matrix_g1(q, omega, m_chi, v_e):
    """
    Computes the g1 integral for each (q, omega) pair
    See Eq. C11 in EFT paper (2009.13534)

    The result is a 3D array of shape (n_q, n_modes, 3)
    (i.e., for each q-point and mode, we get a 3-vector)

    TODO: this is a little off, fix it
    """

    g0 = matrix_g0(q, omega, m_chi, v_e)
    qhat = q / np.linalg.norm(q, axis=1)[:, None]

    matrix1 = (omega / np.linalg.norm(q, axis=1)[:, None])[:, :, None] * qhat[:, None]
    matrix2 = (np.eye(3) - np.einsum("ij,ik->ijk", qhat, qhat)) @ v_e
    return (matrix1 - matrix2[:, None, :]) * g0[:, :, None]
