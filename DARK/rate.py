import math

import numpy as np
import numpy.linalg as LA
from numpy.typing import ArrayLike
from DARK.core import MagnonMaterial, Model, Numerics
import DARK.constants as const
from DARK.velocity_g import matrix_g0

from DARK.grids import (
    create_q_mesh,
    generate_k_XYZ_mesh_from_q_XYZ_mesh,
    get_G_XYZ_list_from_q_XYZ_list,
)


def sigma_mdm(q, epsilons):
    # Eq (65) in arXiv:2009.13534
    qhat = q / np.linalg.norm(q, axis=1)[:, None]
    n_q, n_modes = epsilons.shape[0], epsilons.shape[1]
    identity = np.tile(np.eye(3)[None, :, :], (n_q, n_modes, 1, 1))
    matrix = identity - np.tile(
        np.einsum("ij,ik->ijk", qhat, qhat)[:, None, :, :], (1, n_modes, 1, 1)
    )
    sigma = (
        LA.norm(np.matmul(matrix, 2 * const.mu_tilde_e * epsilons[..., None]), axis=-2)
        ** 2
    )
    return sigma[:, :, 0]


def sigma_ap(q, epsilons):
    # Eq (66) in arXiv:2009.13534
    tiled_q = np.tile(q[None, :, :], (epsilons.shape[1], 1, 1)).swapaxes(0, 1)
    return LA.norm(np.cross(tiled_q, 2 * const.mu_tilde_e * epsilons), axis=2) ** 2


class MagnonCalculation:
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

        delta = 2 * model.power_V - 2 * model.Fmed_power
        [self.q_cart, self.jacobian] = create_q_mesh(
            m_chi,
            0,
            self.v_e,
            numerics,
            material,
            delta,
        )

        self.k_cart = generate_k_XYZ_mesh_from_q_XYZ_mesh(
            self.q_cart, material.recip_frac_to_cart
        )
        self.G_cart = get_G_XYZ_list_from_q_XYZ_list(
            self.q_cart, material.recip_frac_to_cart
        )

    def compute_ve(self, t):
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

    def calculate_rate(
        self,
    ):
        """
        Computes the differential rate
        """

        # threshold     = physics_parameters['threshold']
        # m_cell = 2749.367e9 # YIG mass, all ions
        # m_cell = 821.5e9 # For VBTS
        # m_cell = 52.45e9 # YIG, Fe3+ only
        # idk how to set this for magnons tbh, just a max magnon energy * 4
        # max_delta_E = 4 * 90e-3  # Should be a material property
        # max_delta_E = 2 * 30e-3  # For VBTS
        max_bin_num = math.ceil(self.material.max_dE / self.numerics.bin_width)

        n_modes = self.material.n_modes
        n_q = len(self.q_cart)

        diff_rate = np.zeros(max_bin_num, dtype=complex)
        binned_rate = np.zeros(n_modes, dtype=complex)
        omegas = np.zeros((n_q, n_modes))
        epsilons = np.zeros((n_q, n_modes, 3), dtype=complex)

        model_name = self.model.name

        # TODO: implement this without a loop?
        for iq, (G, k) in enumerate(zip(self.G_cart, self.k_cart)):
            if iq % 1000 == 0:
                print(f"* m_chi = {self.m_chi:13.4f}, q-point: {iq:6d}/{n_q:6d})")
            omegas[iq, :], epsilons[iq, :, :] = self.material.get_eig(k, G)

        # Along with omega and epsilons, these are all q*nu arrays
        bin_num = np.floor((omegas) / self.numerics.bin_width).astype(int)
        g0 = matrix_g0(self.q_cart, omegas, self.m_chi, self.v_e)
        if model_name == "mdm":
            sigma_nu_q = sigma_mdm(self.q_cart, epsilons)
        elif model_name == "ap":
            sigma_nu_q = sigma_ap(self.q_cart, epsilons)
        tiled_jacobian = np.tile(self.jacobian, (n_modes, 1)).T

        # Integrate to get deltaR
        vol_element = tiled_jacobian * (
            (2 * np.pi) ** 3 * np.prod(self.numerics.N_abc)
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
