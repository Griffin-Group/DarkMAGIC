import itertools
from typing import Tuple

import numba as nb
import numpy as np
from numba import njit, prange
from numba.experimental import jitclass
from numpy import linalg as LA
from numpy.typing import ArrayLike

import darkmagic.constants as const


@jitclass(
    [
        ("k_frac", nb.float64[:, :]),
        ("weights", nb.int64[:]),
    ]
)
class MonkhorstPackGrid:
    """
    A class representing a Monkhorst-Pack grid for Brillouin zone sampling.

    Attributes:
        k_frac (ndarray): The fractional coordinates of the k vectors.
        weights (ndarray): The weights of the k vectors.
    """

    def __init__(
        self,
        N_grid: np.ndarray,
        shift: bool = True,
    ):
        """
        Constructor for the MonkhorstPackGrid class.

        N_grid (ArrayLike): The number of grid points along each reciprocal lattice vector.
        shift (bool): Whether to shift the grid. Defaults to a shifted grid to avoid singularities in DWF.
        """
        s = np.array([1, 1, 1]) / 2 if shift else np.array([0, 0, 0])
        grid_indices = np.meshgrid(*[np.arange(n) for n in N_grid], indexing="ij")
        k_list = np.stack(grid_indices, axis=-1).reshape(-1, 3) / N_grid
        k_list += ((N_grid + 1) % 2 - N_grid // 2 + s) / N_grid
        self.k_frac = np.array(sorted(k_list, key=tuple), dtype=np.float64)
        self.weights = np.ones_like(self.k_frac[:, 0], dtype=np.int64)
        # SGA struggles with the struct in 1/eV, so we scale back to ang
        # struct = deepcopy(material.structure)
        # struct.scale_lattice(struct.volume * (const.inveV_to_Ang) ** 3)
        # sga = SpacegroupAnalyzer(struct)
        # if use_sym:
        #     points = sga.get_ir_reciprocal_mesh(N_grid, is_shift=shift)
        #     self.k_frac, self.weights = map(np.array, zip(*points))
        # else:
        #     self.k_frac, _ = sga.get_ir_reciprocal_mesh_map(N_grid, is_shift=shift)
        #     self.weights = np.ones_like(self.k_frac[:, 0])


@jitclass(
    [
        ("nq", nb.int64),
        ("q_max", nb.float64),
        ("q_cart", nb.float64[:, :]),
        ("q_frac", nb.float64[:, :]),
        ("q_norm", nb.float64[:]),
        ("q_hat", nb.float64[:, :]),
        ("G_cart", nb.float64[:, :]),
        ("G_frac", nb.int64[:, :]),
        ("k_frac", nb.float64[:, :]),
        ("k_cart", nb.float64[:, :]),
        ("jacobian", nb.float64[:]),
        ("vol_element", nb.float64[:]),
        ("_qhat_qhat", nb.float64[:, :, :]),
    ]
)
class SphericalGrid:
    """
    Represents a spherical grid used for numerical calculations in DarkMAGIC.


    Attributes:
        nq (int): The number of q points.
        q_max (float): The maximum value of the q vector (eV).
        q_cart (ndarray): The Cartesian coordinates of the q vectors (eV).
        q_frac (ndarray): The fractional coordinates of the q vectors.
        q_norm (ndarray): The norms of the q vectors (eV).
        q_hat (ndarray): The unit vectors of the q vectors.
        G_cart (ndarray): The Cartesian coordinates of the G vectors (eV).
        G_frac (ndarray): The fractional coordinates of the G vectors.
        k_frac (ndarray): The fractional coordinates of the k vectors.
        k_cart (ndarray): The Cartesian coordinates of the k vectors.
        vol_element (ndarray): Volume element for the grid.
    """

    def __init__(
        self,
        m_chi: float,
        v_e: ArrayLike,
        q_cut: float,
        N_grid: ArrayLike,
        recip_cart_to_frac: np.ndarray,
    ):
        """
        Spherical grid constructor.

        Args:
            m_chi: The mass of the dark matter particle.
            v_e: The velocity of the Earth.
            q_cut: cutoff value for the momentum transfer, in eV
            N_grid: The number of grid points (radial, azimuthal, polar)
            recip_cart_to_frac: The k-space transformation matrix from Cartesian in eV to fractional coordinates.
        """
        # Get q and G vectors
        self.q_max = min(2 * m_chi * (const.VESC + const.VE), q_cut)
        self.q_cart, self.q_frac = self._get_q_points(
            N_grid, m_chi, v_e, recip_cart_to_frac
        )
        self.nq = self.q_cart.shape[0]
        # These show up often so it's efficient to compute them only once
        self.q_norm = LA.norm(self.q_cart, axis=1)
        self.q_hat = self.q_cart / self.q_norm[:, None]

        # Get G-vectors
        recip_frac_to_cart = LA.inv(recip_cart_to_frac)
        self.G_cart, self.G_frac = self._get_G_vectors(recip_frac_to_cart)
        # Deriving this is straightforward, remember we're sampling
        # with a power of 2 in the q direction, hence the square roots on |q|
        jacobian = 8 * np.pi * self.q_norm ** (5 / 2) * self.q_max ** (1 / 2)
        # Volume element dV = d^3q J(q) / (2pi)^3 / N^3
        self.vol_element = jacobian / ((2 * np.pi) ** 3 * np.prod(N_grid))

        # Get the k-vectors
        self.k_frac = self.q_frac - self.G_frac
        self.k_cart = np.matmul(self.k_frac, recip_frac_to_cart)

        # Outer product of qhat with itself is frequently used
        self._qhat_qhat = None

    @property
    def qhat_qhat(self):
        if self._qhat_qhat is None:
            self._qhat_qhat = np.einsum("...i,...j->...ij", self.q_hat, self.q_hat)
        return self._qhat_qhat

    def _get_q_points(
        self,
        N_grid: ArrayLike,
        m_chi: float,
        v_e: ArrayLike,
        recip_cart_to_frac: ArrayLike,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the q points for the spherical grid.

        Args:
            N_grid (tuple): A tuple containing the number of points in each direction.
            m_chi (float): The mass of the DM particle.
            v_e (float): The velocity of the Earth in natural units
            recip_cart_to_frac (ndarray): The k-space transformation matrix from Cartesian in Angstroms to fractional coordinates.

        Returns:
            tuple: A tuple containing the Cartesian and fractional coordinates of the q points.
        """
        # Get maximum q value and generate spherical grid
        q_cart = self._generate_spherical_grid(self.q_max, *N_grid)

        # Apply kinematic constraints
        q_norm = LA.norm(q_cart, axis=1)
        with np.errstate(invalid="ignore"):
            q_cart = q_cart[
                (np.abs(np.dot(q_cart, v_e) / q_norm + q_norm / 2 / m_chi) < const.VESC)
            ]
        # TODO: double check that right multiplication works well
        q_frac = np.matmul(q_cart, recip_cart_to_frac)

        return q_cart, q_frac

    def _get_G_vectors(
        self, recip_frac_to_cart: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the G vectors for each q-point, so that each q can be written as
        $\bm{q} = \bm{k} + \bm{G}$, for some k in the first Brillouin zone.

        Args:
            recip_cart_to_frac: The k-space transformation matrix from fractional to cartesian coordinates in Angstroms.

        Returns:
            A tuple containing the Cartesian and fractional coordinates of the G vectors.
        """
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
    def _generate_spherical_grid(radius, n_r, n_phi, n_theta) -> np.ndarray:
        """
        Generate a spherical grid inside and up to the surface of a sphere of radius `radius`
        with `n_r` points in the radial direction, `n_theta` points in the polar direction,
        and `n_phi` points in the azimuthal direction.

        Args:
            radius (float): The radius of the sphere.
            n_r (int): The number of points in the radial direction.
            n_phi (int): The number of points in the azimuthal direction.
            n_theta (int): The number of points in the polar direction.

        Returns:
            ndarray: An array containing the Cartesian coordinates of the points in the spherical grid.
        """
        # Using physicists' convention for theta and phi
        delta_r = 1 / (2 * n_r)
        # a = np.linspace(delta_r, 1 - delta_r, n_r)
        r = radius * np.linspace(delta_r, 1 - delta_r, n_r) ** 2
        # print(np.sqrt(r/radius), r)
        delta_phi = 1 / (2 * n_phi)
        # c = np.linspace(delta_phi, 1 - delta_phi, n_phi)
        phi = 2 * np.pi * np.linspace(delta_phi, 1 - delta_phi, n_phi)  # Azimuthal
        delta_theta = 1 / (2 * n_theta)
        # theta = np.pi * np.linspace(delta_theta, 1 - delta_theta, n_theta)  # Polar
        b = np.linspace(delta_theta, 1 - delta_theta, n_theta)
        theta = np.arccos(2 * b - 1)

        # Convert to cartesian
        x = np.outer(r, np.outer(np.sin(theta), np.cos(phi))).flatten()
        y = np.outer(r, np.outer(np.sin(theta), np.sin(phi))).flatten()
        z = np.outer(r, np.outer(np.cos(theta), np.ones_like(phi))).flatten()

        return np.array([x, y, z]).T


@jitclass(
    [
        ("N_grid", nb.optional(nb.int64[:])),
        ("_power_abc", nb.optional(nb.int64[:])),
        ("N_DWF_grid", nb.optional(nb.int64[:])),
        ("bin_width", nb.float64),
        ("use_q_cut", nb.boolean),
        ("_use_special_mesh", nb.boolean),
        ("_threshold", nb.float64),
    ]
)
class Numerics:
    r"""
    A class that represents the numerical parameters for DarkMAGIC calculations.

    Attributes:
        N_grid (ndarray): The number of grid points in the spherical grid used to sample the momentum transfer (radial, azimuthal, polar).
        power_abc (ndarray): The power of each dimension in the grid (currently unsued)
        N_DWF_grid (ndarray): The size of the Monkhorst-Pack grid used to compute the Debye-Waller factor.
        bin_width (float): The width of the energy bin. Rebinning to larger bins is possible in postprocessing.
        use_q_cut (bool): Whether to use a cutoff for the momentum transfer from DM. If False, the cutoff is set to the maximum possible value $2 m_{\chi} (v_{\text{esc}} + v_{\text{e}})$.
        use_special_mesh (bool): Whether to use a special mesh for the spherical grid (currently unused)

    Methods:
        get_grid(m_chi: float, v_e: ArrayLike, q_cut: float, recip_cart_to_frac: np.ndarray) -> SphericalGrid:
            Returns the spherical grid for the given dark matter mass, and earth velocity.

        get_DWF_grid() -> MonkhorstPackGrid:
            Returns the Monkhorst-Pack grid for computing the Debye-Waller factor.
    """

    def __init__(
        self,
        N_grid: ArrayLike = None,
        N_DWF_grid: ArrayLike = None,
        bin_width: float = 1e-3,
        use_q_cut: bool = True,
        use_special_mesh: bool = False,
        threshold=0,
        power_abc: ArrayLike = None,
    ):
        r"""
        Constructor for the Numerics class.

        Args:
            N_grid (ndarray): The number of grid points in the spherical grid used to sample the momentum transfer (radial, azimuthal, polar).
            power_abc (ndarray): The power of each dimension in the grid (currently unsued)
            N_DWF_grid (ndarray): The size of the Monkhorst-Pack grid used to compute the Debye-Waller factor.
            bin_width (float): The width of the energy bin. Rebinning to larger bins is possible in postprocessing.
            use_q_cut (bool): Whether to use a cutoff for the momentum transfer from DM. If False, the cutoff is set to the maximum possible value $2 m_{\chi} (v_{\text{esc}} + v_{\text{e}})$.
            use_special_mesh (bool): Whether to use a special mesh for the spherical grid (currently unused)
            threshold (float): unused, DarkMAGIC does not impose any thresholds during the actual calculation. Maintained for backwards compatibility.
        """
        self.N_grid = (
            np.array(N_grid, dtype=np.int64)
            if N_grid is not None
            else np.array([20, 10, 10], dtype=np.int64)
        )
        self._power_abc = (
            np.array(power_abc, dtype=np.int64)
            if power_abc is not None
            else np.array([2, 1, 1], dtype=np.int64)
        )
        self.N_DWF_grid = (
            np.array(N_DWF_grid, dtype=np.int64)
            if N_DWF_grid is not None
            else np.array([20, 20, 20], dtype=np.int64)
        )
        self.bin_width = bin_width
        self.use_q_cut = use_q_cut
        self._use_special_mesh = use_special_mesh
        self._threshold = threshold

    # NOTE: numba doesn't support class methods :(
    # @classmethod
    # def from_dict(cls, d: dict):
    #    """
    #    Create a Numerics object from a dictionary.

    #    Args:
    #        d (dict): The dictionary containing the numerical parameters.

    #    Returns:
    #        Numerics: The Numerics object.
    #    """
    #    return cls(
    #        d["N_grid"],
    #        d["N_DWF_grid"],
    #        d["bin_width"],
    #        d["use_q_cut"],
    #        d["_use_special_mesh"],
    #        d["_threshold"],
    #        d["_power_abc"],
    #    )

    def to_dict(self):
        """
        Convert the Numerics object to a dictionary.

        Returns:
            dict: The dictionary containing the numerical parameters.
        """
        return {
            "N_grid": self.N_grid,
            "N_DWF_grid": self.N_DWF_grid,
            "bin_width": self.bin_width,
            "use_q_cut": self.use_q_cut,
            "_use_special_mesh": self._use_special_mesh,
            "_threshold": self._threshold,
            "_power_abc": self._power_abc,
        }

    def get_grid(
        self, m_chi: float, v_e: ArrayLike, q_cut: float, recip_cart_to_frac: np.ndarray
    ) -> SphericalGrid:
        """
        Returns the spherical grid object.

        Args:
            m_chi (float): The mass of the dark matter particle.
            v_e (ArrayLike): The velocity of the Earth.
            q_cut (float): The cutoff value for the momentum transfer.
            recip_cart_to_frac (ndarray): The k-space transformation matrix from Cartesian in eV to fractional coordinates.
        Returns:
            SphericalGrid: The spherical grid object.
        """
        q_cut = q_cut if self.use_q_cut else 1e10
        return SphericalGrid(m_chi, v_e, q_cut, self.N_grid, recip_cart_to_frac)

    def get_DWF_grid(self):  # -> MonkhorstPackGrid:
        """
        Returns the density-weighted Fermi grid object.

        Returns:
            MonkhorstPackGrid: The density-weighted Fermi grid object.
        """
        return MonkhorstPackGrid(self.N_DWF_grid, shift=True)


def numerics_from_dict(d: dict):
    """
    Create a Numerics object from a dictionary.
    Note that @classmethod is not currently supported by numba
    hence this annoying workaround.

    Args:
        d (dict): The dictionary containing the numerical parameters.

    Returns:
        Numerics: The Numerics object.
    """
    return Numerics(
        d["N_grid"],
        d["N_DWF_grid"],
        d["bin_width"],
        d["use_q_cut"],
        d["_use_special_mesh"],
        d["_threshold"],
        d["_power_abc"],
    )
