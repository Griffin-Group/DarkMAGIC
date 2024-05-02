import itertools
import math
import warnings
from abc import ABC, abstractmethod

import numpy as np
import numpy.linalg as LA
from numpy.typing import ArrayLike

import darkmagic.constants as const
from darkmagic.benchmark_models import BUILT_IN_MODELS
from darkmagic.io import read_h5
from darkmagic.material import MagnonMaterial, Material, PhononMaterial
from darkmagic.model import Model, Potential
from darkmagic.numerics import Numerics
from darkmagic.v_integrals import MBDistribution

# dictionary to hold the calculation classes
global calc_classes


class FullCalculation:
    """
    A class for calculating rates at a given list of masses and earth velocities.
    """

    def __init__(
        self,
        m_chi: float,
        material: Material,
        model: Model,
        numerics: Numerics,
        time: ArrayLike | None = None,
        v_e: ArrayLike | None = None,
    ):
        if time is None and v_e is None:
            raise ValueError("Either time or v_e must be provided")
        if time is not None and v_e is not None:
            raise ValueError("Only one of time or v_e should be provided")

        self.v_e = self.compute_ve(time) if time is not None else v_e
        self.time = time
        self.numerics = numerics
        self.material = material
        self.m_chi = m_chi
        self.model = model

        # TODO: this is ugly
        calc_class = calc_classes.get(type(material))
        if calc_class is None:
            warnings.warn(
                "Material class not recognized. This is possibly because you"
                " are trying to read in a file that lacks material "
                "information (e.g., PhonoDark formatted files). "
                "If this is not the case, please report a bug."
            )

        self.calc_list = None
        if calc_class is not None:
            self.calc_list = [
                [calc_class(m, v, material, model, numerics) for m in self.m_chi]
                for v in self.v_e
            ]

    @classmethod
    def from_file(cls, filename: str, format="phonodark"):
        """
        Load the rates from a file
        """
        numerics, particle_physics, rates = read_h5(filename, format)
        time = particle_physics["times"]
        m_chi = particle_physics["dm_properties"]["mass_list"]
        numerics = Numerics.from_dict(numerics)
        material = None
        if material is None:
            warnings.warn(
                "Material information not found in file. Running the calculation won't work, but the read in results can be analyzed."
            )
        model = Model.from_dict(particle_physics)

        calc = cls(m_chi, material, model, numerics, time=time)
        calc.binned_rate = rates[0]
        calc.diff_rate = rates[1]
        calc.total_rate = rates[2]

        return calc

    def calculate_rate(self):
        """
        Computes the differential rate for all masses and earth velocities
        """
        max_bin_num = math.ceil(self.material.max_dE / self.numerics.bin_width)
        self.diff_rate = np.zeros((len(self.time), len(self.m_chi), max_bin_num))
        self.binned_rate = np.zeros(
            (len(self.time), len(self.m_chi), self.material.n_modes)
        )
        self.total_rate = np.zeros((len(self.time), len(self.m_chi)))

        for im, iv in itertools.product(range(len(self.m_chi)), range(len(self.time))):
            (
                self.diff_rate[iv, im],
                self.binned_rate[iv, im],
                self.total_rate[iv, im],
            ) = self.calc_list[iv][im].calculate_rate()

    def compute_ve(self, times: float):
        """
        Returns the earth's velocity in the lab frame at time t (in hours)
        """
        theta = const.theta_earth

        v_e = np.zeros((len(times), 3))
        for it, t in enumerate(times):
            phi = 2 * np.pi * (t / 24.0)
            v_e[it] = const.VE * np.array(
                [
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta) * np.sin(theta) * (np.cos(phi) - 1),
                    (np.sin(theta) ** 2) * np.cos(phi) + np.cos(theta) ** 2,
                ]
            )
        return v_e

    def get_reach(
        self,
        threshold_meV: float = 1.0,
        exposure_kg_yr: float = 1.0,
        n_cut: float = 3.0,
        model: str | None = None,
        time: float | str = 0,
    ) -> np.array:
        """
        Computes the projected reach: the 95% C.L. constraint (3 events, no background) on $\bar{sigma}_n$ or $\bar{sigma}_e$ for a given model, in units of cm2 and normalizing to the appropriate reference cross section.

        Args:
            filename (str): The path to the HDF5 file containing the data.
            threshold_meV (float, optional): The threshold in meV. Defaults to 1.0.
            exposure_kg_yr (float, optional): The exposure in kg.yr. Defaults to 1.0.
            n_cut (float, optional): The number of events. Defaults to 3.0.
            model (str, optional): The model to use. Defaults to None (i.e., finds the model name in the file)
            time (float | str, optional): The time to use. Defaults to 0.

        Returns:
            np.array: the cross section.
        """

        energy_bin_width = self.numerics.bin_width
        threshold = self.numerics._threshold  # legacy for PD support
        print("got a threshold of ", threshold)
        m_chi = self.m_chi
        # get time index
        try:
            t_idx = np.argwhere(self.time == time)[0][0]
        except IndexError as e:
            raise ValueError(f"Time {time} not found in the list of times.") from e

        raw_binned_rate = 1e-100 + self.diff_rate[t_idx]  # TODO: bad names...

        # Vanilla PhonoDark calcs have a threshold of 1 meV by default
        # We need to account for that in the binning
        bin_cutoff = int(threshold_meV * 1e-3 / energy_bin_width) - int(
            threshold / energy_bin_width
        )

        model_name = model if model is not None else self.model.shortname
        if model_name is None:
            raise ValueError(
                "Model not provided and not found in the file. "
                "Please provide the model name."
            )
        sigma = n_cut / np.sum(raw_binned_rate[:, bin_cutoff:], axis=1)
        sigma /= (
            exposure_kg_yr
            * const.kg_yr
            * const.cm2
            * BUILT_IN_MODELS[model_name].ref_cross_sect(m_chi)
        )

        return sigma


class Calculation(ABC):
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
        self.v_e = v_e  # self.compute_ve(time) if time is not None else v_e
        self.material = material
        self.model = model
        self.numerics = numerics
        self.grid = numerics.get_grid(m_chi, self.v_e, material)
        self.dwf_grid = numerics.get_DWF_grid(material)

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
        bin_num = np.floor((omegas) / self.numerics.bin_width).astype(int)
        # Each bin has the rate from processes of energies within that bin
        diff_rate = np.zeros(max_bin_num)
        np.add.at(diff_rate, bin_num, deltaR)
        # Integrate over q to get rate from different modes
        binned_rate = np.sum(deltaR, axis=0)
        # Sum over modes to get total rate
        total_rate = np.sum(diff_rate)

        return [diff_rate, binned_rate, total_rate]


class MagnonCalculation(Calculation):
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


class PhononCalculation(Calculation):
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

        # H(q)_{\nu j} = e^{i G x_j} e^{- W_j(q)}  \times
        # \frac{q \cdot \epsilon_{k j \nu}^*}{\sqrt{2 m_j \omega_{k \nu}}
        H_q_nu_j = (exponential[:, None, :] * q_dot_epsconj) / np.sqrt(
            2 * self.material.m_atoms[None, None, :] * omegas[..., None]
        )
        # TODO: better way to deal with this.
        # We get issues from the very small negative frequencies very close to Gamma
        # Which we're not avoiding since I don't want to put a built-in threshold.
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

        # Get MB distribution
        v_dist = MBDistribution(self.grid, omegas, self.m_chi, self.v_e)

        return sigma0_q_nu * v_dist.G0


calc_classes = {
    PhononMaterial: PhononCalculation,
    MagnonMaterial: MagnonCalculation,
}
