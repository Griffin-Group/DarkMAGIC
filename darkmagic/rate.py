""" "
Module with classes for calculating the rate
"""

import math
from abc import ABC, abstractmethod

import numpy as np
import numpy.linalg as LA
from numpy.typing import ArrayLike

import darkmagic.constants as const
from darkmagic.material import MagnonMaterial, Material, PhononMaterial
from darkmagic.model import Model, Potential
from darkmagic.numerics import Numerics
from darkmagic.maxwell_boltzmann import MBDistribution

# dictionary to hold the calculation classes
global RATE_CALC_CLASSES


class SingleRateCalc(ABC):
    """
    An abstract class for calculating rates at a given mass and earth velocity
    """

    def __init__(
        self,
        m_chi: float,
        v_e: ArrayLike,
        material: Material,
        model: Model,
        numerics: Numerics,
    ):
        self.m_chi = m_chi
        self.v_e = v_e
        self.material = material
        self.model = model
        self.numerics = numerics
        self._grid = None
        self._dwf_grid = None

    @property
    def grid(self):
        if self._grid is None:
            self._grid = self.numerics.get_grid(self.m_chi, self.v_e, self.material)
        return self._grid

    @property
    def dwf_grid(self):
        if self._dwf_grid is None:
            self._dwf_grid = self.numerics.get_DWF_grid()
        return self._dwf_grid

    @abstractmethod
    def calculate_sigma_q_nu(self, omegas, epsilons):
        pass

    def calculate_rate(
        self,
    ):
        """
        Computes the differential rate
        """

        # (nq, nmodes) and (nq, nmodes, natoms, 3)
        omegas, epsilons = self.material.get_eig(self.grid)

        # Compute \Sigma_{\nu}(q)
        sigma_q_nu = self.calculate_sigma_q_nu(omegas, epsilons)

        # Integrate to get deltaR
        deltaR = (
            (1 / self.material.m_cell)
            * (const.rho_chi / self.m_chi)
            * self.grid.vol_element[:, None]
            * self.model.F_med_prop(self.grid)[:, None] ** 2
            * sigma_q_nu
        )

        # TODO: the names here aren't great
        max_bin_num = math.ceil(self.material.max_dE / self.numerics.bin_width)
        bin_num = np.floor(omegas / self.numerics.bin_width).astype(int)
        # Each bin has the rate from processes of energies within that bin
        diff_rate = np.zeros(max_bin_num)
        np.add.at(diff_rate, bin_num, deltaR)
        # Integrate over q to get rate from different modes
        binned_rate = np.sum(deltaR, axis=0)
        # Sum over modes to get total rate
        total_rate = np.sum(diff_rate)

        return [diff_rate, binned_rate, total_rate]


class MagnonScatterRate(SingleRateCalc):
    """
    Class for calculating the differential rate for magnon scattering
    """

    def calculate_sigma_q_nu(self, omegas, epsilons):
        # Calculate the prefactor
        prefactor = np.sqrt(self.material.Sj / 2)[None, :] * np.exp(
            1j * np.dot(self.grid.G_cart, self.material.xj.T)
        )  # (nq, na)
        # \bm{E}(q)_{nu} = \sum_j e^{i G \cdot x_j} \sqrt{S_j/2} \bm{\epsilon}_{k \nu j}
        E_q_nu = np.sum(prefactor[:, None, :, None] * epsilons, axis=2)  # (nq, nm, 3)
        if self.model.name == "Magnetic Dipole":
            sigma_nu_q = self.sigma_mdm(self.grid.q_cart, E_q_nu)
        elif self.model.name == "Anapole":
            sigma_nu_q = self.sigma_ap(self.grid.q_cart, E_q_nu)
        else:
            raise ValueError(
                f"Unknown model: {self.model.name}. Generic magnon models not yet implemented, only mdm and ap."
            )

        v_dist = MBDistribution(self.grid, omegas, self.m_chi, self.v_e)
        return sigma_nu_q * v_dist.G0

    @staticmethod
    def sigma_mdm(q, E_q_nu):
        # Eq (65) in arXiv:2009.13534
        qhat = q / np.linalg.norm(q, axis=1)[:, None]
        n_q, n_modes = E_q_nu.shape[0], E_q_nu.shape[1]
        identity = np.tile(np.eye(3)[None, :, :], (n_q, n_modes, 1, 1))
        id_minus_qq = identity - np.tile(
            np.einsum("ij,ik->ijk", qhat, qhat)[:, None, :, :], (1, n_modes, 1, 1)
        )
        sigma = (
            LA.norm(
                np.matmul(id_minus_qq, 2 * const.mu_tilde_e * E_q_nu[..., None]),
                axis=-2,
            )
            ** 2
        )
        return sigma[:, :, 0]

    @staticmethod
    def sigma_ap(q, E_q_nu):
        # Eq (66) in arXiv:2009.13534
        tiled_q = np.tile(q[None, :, :], (E_q_nu.shape[1], 1, 1)).swapaxes(0, 1)
        return LA.norm(np.cross(tiled_q, 2 * const.mu_tilde_e * E_q_nu), axis=2) ** 2


class PhononScatterRate(SingleRateCalc):
    """
    Class for calculating the differential rate for phonon scattering
    """

    def calculate_sigma_q_nu(self, omegas, epsilons):
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
        xj = self.material.xj
        G = self.grid.G_cart
        exponential = np.exp(1j * np.dot(G, xj.T) - W_q_j)  # (nq, na)

        # q_\alpha \epsilon_{k \nu j \alpha}
        q_dot_epsconj = np.sum(
            self.grid.q_cart[:, None, None, :] * epsilons.conj(), axis=3
        )  # (nq, nmodes, na)

        # NOTE: we have some very small negative frequencies near Gamma
        # due to numerical errors in applying the ASR. This is normal but
        # Cause issues due to the 1/sqrt(omega) factor. This can be avoided
        # With a built-in threshold but I want to avoid that for now.
        # We just filter these NaNs out by setting them to zeros so they
        # don't contribute to the rate.
        # TODO: think of a better way to handle this
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in")
            # H(q)_{\nu j} = e^{i G x_j} e^{- W_j(q)}  \times
            # \frac{q \cdot \epsilon_{k j \nu}^*}{\sqrt{2 m_j \omega_{k \nu}}
            H_q_nu_j = (exponential[:, None, :] * q_dot_epsconj) / np.sqrt(
                2 * self.material.m_atoms[None, None, :] * omegas[..., None]
            )
        H_q_nu_j = np.nan_to_num(H_q_nu_j)  # Set nans to zeros

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

        # Get MB distribution
        v_dist = MBDistribution(self.grid, omegas, self.m_chi, self.v_e)

        return sigma0_q_nu * v_dist.G0


RATE_CALC_CLASSES = {
    ("scattering", PhononMaterial): PhononScatterRate,
    ("scattering", MagnonMaterial): MagnonScatterRate,
}
