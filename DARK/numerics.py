import itertools

import numpy as np
from numpy.typing import ArrayLike
from numpy import linalg as LA
from pymatgen.core.structure import Structure

import DARK.constants as const


class Numerics:
    def __init__(
        self,
        N_grid: ArrayLike = [20, 10, 10],
        power_abc: ArrayLike = [2, 1, 1],
        n_DW_xyz: ArrayLike = [20, 20, 20],
        bin_width: float = 1e-3,
        use_q_cut: bool = True,
        use_special_mesh: bool = False,
    ):
        self.N_grid = np.array(N_grid)
        self.power_abc = np.array(power_abc)
        self.n_DW_xyz = np.array(n_DW_xyz)
        self.bin_width = bin_width
        self.use_q_cut = use_q_cut
        self.use_special_mesh = use_special_mesh
        self._grid = None

    def get_grid(self, m_chi, v_e, material):
        if self._grid is None:
            self._grid = Grid(m_chi, v_e, self, material)
        return self._grid


class Grid:
    def __init__(self, m_chi, v_e, numerics, material):
        # Get q and G vectors
        q_cut = material.q_cut if numerics.use_q_cut else 1e10
        self.q_max = min(2 * m_chi * (const.VESC + const.VE), q_cut)
        self.q_cart, self.q_frac = self._get_q_points(
            numerics.N_grid, m_chi, v_e, material.recip_cart_to_frac
        )
        # These show up in many so it's efficient to compute them only once
        self.q_norm = LA.norm(self.q_cart, axis=1)
        self.q_hat = self.q_cart / self.q_norm[:, None]

        # Get G-vectors
        self.G_cart, self.G_frac = self._get_G_vectors(material.recip_frac_to_cart)
        # Deriving this is straightforward, remember we're sampling
        # with a power of 2 in the q direction, hence the square roots on |q|
        self.jacobian = 8 * np.pi * self.q_norm ** (5 / 2) * self.q_max ** (1 / 2)

        # Get the k-vectors
        self.k_frac = self.q_frac - self.G_frac
        self.k_cart = np.matmul(self.k_frac, material.recip_frac_to_cart)

    def _get_q_points(self, N_grid, m_chi, v_e, recip_cart_to_frac):
        # Get maximum q value and generate spherical grid
        q_cart = self._generate_spherical_grid(self.q_max, *N_grid)

        # Apply kinematic constraints
        q_norm = LA.norm(q_cart, axis=1)
        with np.errstate(invalid="ignore"):
            q_cart = q_cart[
                (np.abs(np.dot(q_cart, v_e) / q_norm + q_norm / 2 / m_chi) < const.VESC)
                & (q_norm > 0)
            ]
        # TODO: double check that right multiplication works well
        q_frac = np.matmul(q_cart, recip_cart_to_frac)

        return q_cart, q_frac

    def _get_G_vectors(self, recip_frac_to_cart):
        # Generate the 8 closest G-vectors to each q-point
        G_frac = np.array(
            [
                list(itertools.product(*zip(np.floor(q), np.ceil(q))))
                for q in self.q_frac
            ]
        )
        # TODO: double check that right multiplication works well
        G_cart = np.matmul(G_frac, recip_frac_to_cart)
        # Compute the distances between the q-point and each of the 8 G-vectors
        dist = LA.norm(G_cart - self.q_cart[:, None, :], axis=-1)
        # Pick out the nearest G-vector for each q-point
        n_q = self.q_frac.shape[0]
        G_cart = G_cart[np.arange(n_q), mask := np.argmin(dist, axis=1)]
        G_frac = G_frac[np.arange(n_q), mask]

        return G_cart, G_frac

    @staticmethod
    def _generate_spherical_grid(radius, n_r, n_phi, n_theta):
        """
        Generate a spherical grid inside and up to the surface of a sphere of radius `radius`
        with `n_r` points in the radial direction, `n_theta` points in the polar direction,
        and `n_phi` points in the azimuthal direction.
        Args:
            radius: float, radius of the sphere
            n_r: int, number of points in the radial direction
            n_phi: int, number of points in the azimuthal direction
            n_theta: int, number of points in the polar direction
        """
        # Using physicists' convention for theta and phi
        delta_r = 1 / (2 * n_r)
        a = np.linspace(delta_r, 1 - delta_r, n_r)
        r = radius * np.linspace(delta_r, 1 - delta_r, n_r) ** 2
        # print(np.sqrt(r/radius), r)
        delta_phi = 1 / (2 * n_phi)
        c = np.linspace(delta_phi, 1 - delta_phi, n_phi)
        phi = 2 * np.pi * np.linspace(delta_phi, 1 - delta_phi, n_phi)  # Azimuthal
        delta_theta = 1 / (2 * n_theta)
        # theta = np.pi * np.linspace(delta_theta, 1 - delta_theta, n_theta)  # Polar
        b = np.linspace(delta_theta, 1 - delta_theta, n_theta)
        theta = np.arccos(2 * b - 1)

        # Convert to cartesian
        x = np.outer(r, np.outer(np.sin(theta), np.cos(phi))).flatten()
        y = np.outer(r, np.outer(np.sin(theta), np.sin(phi))).flatten()
        z = np.outer(r, np.outer(np.cos(theta), np.ones_like(phi))).flatten()

        qpoints = np.array([x, y, z]).T

        # Get rid of duplicates due to periodicity
        return qpoints  # np.unique(qpoints, axis=0)
