import math

import numpy as np
import numpy.linalg as LA
from numpy.typing import ArrayLike

import darkmagic.constants as const
from darkmagic.material import MagnonMaterial
from darkmagic.model import Model, Potential
from darkmagic.numerics import Numerics
from darkmagic.v_integrals import MBDistribution


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
        self.v_e = self.compute_ve(time) if time is not None else v_e
        self.material = material
        self.model = model
        self.numerics = numerics
        self.grid = numerics.get_grid(m_chi, self.v_e, material)
        self.dwf_grid = numerics.get_DWF_grid(material)

    # TODO: Should take this outside the class and make v_e required
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

        v_dist = MBDistribution(self.grid, omegas, self.m_chi, self.v_e)

        # Along with omega and epsilons, these are all q*nu arrays
        bin_num = np.floor((omegas) / self.numerics.bin_width).astype(int)
        g0 = v_dist.G0

        if model_name == "Magnetic Dipole":
            sigma_nu_q = self.sigma_mdm(self.grid.q_cart, epsilons)
        elif model_name == "Anapole":
            sigma_nu_q = self.sigma_ap(self.grid.q_cart, epsilons)
        else:
            raise ValueError(
                f"Unknown model: {model_name}. Generic magnon models not yet implemented, only mdm and ap."
            )
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


class PhononCalculation(Calculation):
    """
    Class for calculating the differential rate for phonon scattering
    """

    def calculate_rate(
        self,
    ):
        """
        Computes the differential rate
        """

        max_bin_num = math.ceil(self.material.max_dE / self.numerics.bin_width)

        n_modes = self.material.n_modes

        diff_rate = np.zeros(max_bin_num, dtype=complex)
        binned_rate = np.zeros(n_modes, dtype=complex)

        # (nq, nmodes) and (nq, nmodes, natoms, 3)
        omegas, epsilons = self.material.get_eig(self.grid.k_frac)
        # (nq, na, 3, 3)
        W_tensor = self.material.get_W_tensor(self.dwf_grid)

        # W_j(q) = q_\alpha W_\alpha\beta q_\beta (DWF)
        W_q_j = (
            np.sum(
                W_tensor[None, ...] * self.grid.qhat_qhat[:, None, ...], axis=(2, 3)
            ).real
            * self.grid.q_norm[:, None] ** 2
        )
        # exp(i G \cdot x_j - W_j(q))
        xj = self.material.structure.cart_coords
        G = self.grid.G_cart
        exponential = np.exp(1j * np.dot(G, xj.T) - W_q_j)  # (nq, na)

        # q_\alpha \epsilon_{k \nu j \alpha}
        q_dot_epsconj = np.sum(
            self.grid.q_cart[:, None, None, :] * epsilons.conj(), axis=3
        )  # (nq, nmodes, na)

        # H(q)_{\nu j} = e^{i G x_j} e^{- W_j(q)}  \times
        # \frac{q \cdot \epsilon_{k j \nu}^*}{\sqrt{2 m_j \omega_{k \nu}}
        H_q_nu_j = (exponential[:, None, :] * q_dot_epsconj) / np.sqrt(
            2 * self.material.m_atoms[None, None, :] * omegas[..., None]
        )
        # TODO: better way to deal with this.
        # We get issues from the very small negative frequencies very close to Gamma
        # Which we're not avoiding since I don't want to put a build in threshold.
        H_q_nu_j = np.nan_to_num(H_q_nu_j)

        # Compute potential
        pot = Potential(self.model)
        V_q_j = pot.eval_V(self.grid, self.material, self.m_chi, self.model.S_chi)

        # H(q)_{\nu j'}^* \times H(q)_{\nu j}
        Hs_jp_H_j = H_q_nu_j.conj()[..., None] * H_q_nu_j[..., None, :]
        # (V^{00}_{j'}(q))^* \times V^{00}_{j}(q)
        V00s_jp_V00_j = V_q_j["00"].conj()[..., None] * V_q_j["00"][..., None, :]
        # \Sigma^0_{\nu}(q)=\sum_{jj'} H(q)_{\nu j'}^* H(q)_{\nu j} V_{00 j'}^* V_{00 j}
        sigma0_q_nu = np.sum(Hs_jp_H_j * V00s_jp_V00_j[:, None, ...], axis=(2, 3)).real
        # sigma0_q_nu = np.abs(np.sum((H_q_nu_j * V_q_j["00"][:, None, ...]), axis=2))**2

        # Now we need the maxwell boltzmann distribution
        v_dist = MBDistribution(self.grid, omegas, self.m_chi, self.v_e)
        G0 = v_dist.G0

        # Integrate to get deltaR
        tiled_jacobian = np.tile(self.grid.jacobian, (n_modes, 1)).T
        vol_element = tiled_jacobian * (
            (2 * np.pi) ** 3 * np.prod(self.numerics.N_grid)
        ) ** (-1)
        deltaR = (
            (1 / self.material.m_cell)
            * (const.rho_chi / self.m_chi)
            * vol_element
            * self.model.F_med_prop(self.grid)[:, None] ** 2
            * (sigma0_q_nu * G0)
        )
        # Get diff rate, binned rate and total rate
        bin_num = np.floor((omegas) / self.numerics.bin_width).astype(int)
        diff_rate = np.zeros(max_bin_num)
        np.add.at(diff_rate, bin_num, deltaR)
        binned_rate = np.sum(deltaR, axis=0)
        total_rate = np.sum(diff_rate)

        return [diff_rate, binned_rate, total_rate]
