"""
Module for classes and utilities for the Maxwell-Boltzmann distribution and the velocity (G) integrals
"""

import numpy as np
import numpy.linalg as LA

import darkmagic.constants as const


class MBDistribution:
    """
    Class for truncated Maxwell-Boltzmann distribution
    """

    def __init__(self, grid, omega, m_chi, v_e) -> None:
        self.grid = grid
        self.omega = omega
        self.m_chi = m_chi
        self.v_e = np.array(v_e)

        # Internal variables
        self._v_minus = None
        self._G0 = None
        self._G1 = None
        self._G2 = None
        self._F = None
        self._X = None
        self._eye_minus_qhat_qhat = None

    @property
    def v_minus(self):
        """
        Computes v_minus = min(|v_star|, Vesc) for each (q, omega) pair
        q: numpy array of shape (n_q, 3), momentum transfer
        omega: 2D numpy array of shape (n_q, n_modes), eigenmodes
        m_chi: float, DM mass
        v_e: vector, lab frame velocity of earth
        """
        if self._v_minus is None:
            q = self.grid.q_cart
            n_modes = self.omega.shape[1]
            tiled_q = np.tile(q[None, :, :], (n_modes, 1, 1)).swapaxes(0, 1)
            q_norm = LA.norm(tiled_q, axis=2)
            v_star = (
                1
                / q_norm
                * (np.dot(tiled_q, self.v_e) + q_norm**2 / 2 / self.m_chi + self.omega)
            )
            self._v_minus = np.minimum(np.abs(v_star), const.VESC)
        return self._v_minus

    @property
    def G0(self):
        """
        Computes the G0 integral for each (q, omega) pair
        See Eq. C9 in EFT paper (2009.13534)
        """
        if self._G0 is None:
            c1 = 2 * np.pi**2 * const.V0**2 / self.grid.q_norm / const.N0
            c1 = np.tile(c1, (self.omega.shape[1], 1)).T

            self._G0 = c1 * (
                np.exp(-((self.v_minus / const.V0) ** 2))
                - np.exp(-((const.VESC / const.V0) ** 2))
            )
        return self._G0

    @property
    def G1(self):
        r"""
        Computes the g1 integral for each (q, omega) pair. See Eq. C11 in EFT paper (2009.13534). Note that my definition here is slightly different from that equation. See formalism notes.

        TODO: update this docstring with all the equations.

        The result is a 3D array of shape (n_q, n_modes, 3)
        (i.e., for each q-point and mode, we get a 3-vector)

        This integral is equivalent to

        $$
        g1 = (v_star \hat{q} - v_e - \vec{q}/2/m_chi) * g0
        $$

        Note: this is defined slightly differently in the original
              PhonoDark. Specifically,

        $$
        g1 = (v_star R[:,2] - v_e) * g0
        $$

        where R is a matrix that rotates \hat{q} to lie along the z-axis
        """
        if self._G1 is None:
            self._G1 = self.X * self.G0[:, :, None]
        return self._G1

    @property
    def G2(self):
        """
        Computes the g2 integral for each (q, omega) pair. See Eq. C14 in EFT paper (2009.13534). Note that my definition here is slightly different, see the formalism notes.

        TODO: update this with all the equations

        The result is a 4D array of shape (n_q, n_modes, 3, 3)
        (i.e., for each q-point and mode, we get a 3x3 matrix)

        This suffers from a similar problem to g1, the definition
        in PhonoDark is quite different from the paper (including
        the definition of what I call F below) and I don't know
        how to reconcile them. It also invovles a rotation,
        like in g1.
        """
        if self._G2 is None:
            term1 = (
                np.einsum("...i,...j->...ij", self.X, self.X) * self.G0[..., None, None]
            )
            term2 = self.eye_minus_qhat_qhat[:, None, :, :] * self.F[..., None, None]
            self._G2 = term1 + term2
        return self._G2

    @property
    def F(self):
        r"""
        Computes the very last term of Eq. (C14) in the EFT paper (2009.13534)
        This is the term that multiplies (1 - qhat \otimes qhat) in the g2 integral
        """
        if self._F is None:
            v_minus = self.v_minus
            C = np.pi**2 * const.V0**2 / const.N0 / self.grid.q_norm  # nq x 1
            self._F = C[:, None] * (
                const.V0**2 * np.exp(-((v_minus / const.V0) ** 2))
                - (const.V0**2 - v_minus**2 + const.VESC**2)
                * np.exp(-((const.VESC / const.V0) ** 2))
            )
        return self._F

    @property
    def X(self):
        r"""
        Computes the X vector for each (q, omega) pair

        $$
        X = (\omega / |q|) \hat{q} - (\mathbb{1} - \hat{q} \otimes \hat{q}) v_e
        $$

        The result is a 3D array of shape (n_q, n_modes, 3)
        """
        if self._X is None:
            term1 = (self.omega / self.grid.q_norm[:, None])[
                :, :, None
            ] * self.grid.q_hat[:, None]
            term2 = self.eye_minus_qhat_qhat @ self.v_e
            self._X = term1 - term2[:, None, :]
        return self._X

    @property
    def eye_minus_qhat_qhat(self):
        r"""
        Computes the (3x3) matrix (1 - qhat \otimes qhat) for each q-point
        The result is a 3D array of shape (n_q, 3, 3)
        """
        if self._eye_minus_qhat_qhat is None:
            self._eye_minus_qhat_qhat = np.eye(3) - self.grid.qhat_qhat
        return self._eye_minus_qhat_qhat
