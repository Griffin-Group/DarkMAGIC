import math

import numpy as np
import numpy.linalg as LA
from numpy.typing import ArrayLike
from DARK.material import MagnonMaterial
from DARK.model import Model
from DARK.numerics import Numerics
import DARK.constants as const
from DARK.v_integrals import MBDistribution


class Calculation:
    """
    Generic class for calculating the differential rate
    """

    def __init__(
        self,
        m_chi: float,
        material: MagnonMaterial,
        model: Model,
        numerics: Numerics,
        time: float | None = 0,
        v_e: ArrayLike | None = None,
    ):
        self.m_chi = m_chi
        if time is None and v_e is None:
            raise ValueError("Either time or v_e must be provided")
        if time is not None and v_e is not None:
            raise ValueError("Only one of time or v_e should be provided")
        if time is not None:
            self.v_e = self.compute_ve(time)
        else:
            self.v_e = v_e

        self.material = material
        self.model = model
        self.numerics = numerics
        self.grid = numerics.get_grid(m_chi, self.v_e, material)

    def compute_ve(self, t: float):
        """
        Returns the earth's velocity in the lab frame at time t (in hours)
        """
        phi = 2 * np.pi * (t / 24.0)
        theta = const.theta_earth

        return const.VE * np.array(
            [
                np.sin(theta) * np.sin(phi),
                np.cos(theta) * np.sin(theta) * (np.cos(phi) - 1),
                (np.sin(theta) ** 2) * np.cos(phi) + np.cos(theta) ** 2,
            ]
        )


class MagnonCalculation(Calculation):
    """
    Class for calculating the differential rate for magnon scattering
    """

    @staticmethod
    def sigma_mdm(q, epsilons):
        # Eq (65) in arXiv:2009.13534
        qhat = q / np.linalg.norm(q, axis=1)[:, None]
        n_q, n_modes = epsilons.shape[0], epsilons.shape[1]
        identity = np.tile(np.eye(3)[None, :, :], (n_q, n_modes, 1, 1))
        id_minus_qq = identity - np.tile(
            np.einsum("ij,ik->ijk", qhat, qhat)[:, None, :, :], (1, n_modes, 1, 1)
        )
        sigma = (
            LA.norm(
                np.matmul(id_minus_qq, 2 * const.mu_tilde_e * epsilons[..., None]),
                axis=-2,
            )
            ** 2
        )
        return sigma[:, :, 0]

    @staticmethod
    def sigma_ap(q, epsilons):
        # Eq (66) in arXiv:2009.13534
        tiled_q = np.tile(q[None, :, :], (epsilons.shape[1], 1, 1)).swapaxes(0, 1)
        return LA.norm(np.cross(tiled_q, 2 * const.mu_tilde_e * epsilons), axis=2) ** 2

    def calculate_rate(
        self,
    ):
        """
        Computes the differential rate
        """

        max_bin_num = math.ceil(self.material.max_dE / self.numerics.bin_width)

        n_modes = self.material.n_modes
        n_q = len(self.grid.q_cart)

        diff_rate = np.zeros(max_bin_num, dtype=complex)
        binned_rate = np.zeros(n_modes, dtype=complex)
        omegas = np.zeros((n_q, n_modes))
        epsilons = np.zeros((n_q, n_modes, 3), dtype=complex)

        model_name = self.model.name

        # TODO: implement this without a loop?
        for iq, (G, k) in enumerate(zip(self.grid.G_cart, self.grid.k_cart)):
            if iq % 1000 == 0:
                print(f"* m_chi = {self.m_chi:13.4f}, q-point: {iq:6d}/{n_q:6d})")
            omegas[iq, :], epsilons[iq, :, :] = self.material.get_eig(k, G)

        # Along with omega and epsilons, these are all q*nu arrays
        bin_num = np.floor((omegas) / self.numerics.bin_width).astype(int)
        g0 = matrix_g0(self.grid.q_cart, omegas, self.m_chi, self.v_e)
        if model_name == "mdm":
            sigma_nu_q = self.sigma_mdm(self.grid.q_cart, epsilons)
        elif model_name == "ap":
            sigma_nu_q = self.sigma_ap(self.grid.q_cart, epsilons)
        tiled_jacobian = np.tile(self.grid.jacobian, (n_modes, 1)).T

        # Integrate to get deltaR
        vol_element = tiled_jacobian * (
            (2 * np.pi) ** 3 * np.prod(self.numerics.N_grid)
        ) ** (-1)
        deltaR = (
            (1 / self.material.m_cell)
            * (const.rho_chi / self.m_chi)
            * vol_element
            * sigma_nu_q
            * g0
        )

        # Get diff rate, binned rate and total rate
        diff_rate = np.zeros(max_bin_num)
        np.add.at(diff_rate, bin_num, deltaR)
        binned_rate = np.sum(deltaR, axis=0)
        total_rate = sum(diff_rate)

        return [diff_rate, binned_rate, total_rate]
