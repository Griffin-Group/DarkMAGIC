"""
Module for classes used to represent materials and their properties
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import phonopy
from numpy import linalg as LA
from numpy.typing import ArrayLike
from pymatgen.core.structure import Structure
from radtools import MagnonDispersion, SpinHamiltonian

import darkmagic.constants as const
from darkmagic.numerics import SphericalGrid, MonkhorstPackGrid


class MaterialParameters:
    r"""
    Class for DM-relevant material properties, such as the number of fermions, spin, orbital angular momentum, etc.


    Attributes:
        N (dict): Fermion numbers.
        S (dict): Spin vectors.
        L (dict): Orbital angular momentum vectors.
        L_dot_S (dict): $L \cdot S$
        L_tens_S (dict): Spin orbit coupling tensor $L \otimes S$
        lambda_S (ArrayLike): spin-coefficient for magnons
        lambda_L (ArrayLike): orbital angular mom.-coefficient for magnons
        m_psi (dict): Dictionary of masses for different particles.

    Methods:
        validate_for_phonons:
            Validates that the material properties are suitable for phonon calculations.

        validate_for_magnons:
            Validates that the material properties are suitable for magnon calculations.

    """

    def __init__(
        self,
        N: dict = None,
        S: dict = None,
        L: dict = None,
        L_dot_S: dict = None,
        L_tens_S: dict = None,
        lambda_S: ArrayLike = None,
        lambda_L: ArrayLike = None,
        m_psi: dict = None,
    ):
        r"""
        Material properties constructor. All dicts have keys "n", "p", "e" for neutron, proton and electron. Any missing values are instantiated to 0.

        Args:
            N (dict, optional): Fermion numbers.
            S (dict, optional): Spin vectors.
            L (dict, optional): Orbital angular momentum vectors.
            L_dot_S (dict, optional): $L \cdot S$
            L_tens_S (dict, optional): Spin orbit coupling tensor $L \otimes S$
            lambda_S (ArrayLike, optional): spin-coefficient for magnons
            lambda_L (ArrayLike, optional): orbital angular mom.-coefficient for magnons
            m_psi (dict, optional): Masses of the fermions. Defaults to NIST values.
        """
        # Phonons
        self.N = N
        self.S = S
        self.L = L
        self.L_dot_S = L_dot_S
        self.L_tens_S = L_tens_S
        # Magnons
        self.lambda_S = lambda_S
        self.lambda_L = lambda_L
        # Mass of the particles
        self.m_psi = m_psi

    def validate_for_phonons(self, n_atoms: int) -> None:
        """
        Validates that the material properties are suitable for phonons.
        Namely, at least one of N, S, L, L_dot_S or L_tens_S must be defined.

        Args:
            n_atoms (int): Number of atoms in the material.

        Raises:
            AssertionError: If any of the required material properties for phonons are missing or have incorrect dimensions.

        """
        assert any([self.N, self.S, self.L, self.L_dot_S, self.L_tens_S])
        for d in [self.N, self.S, self.L, self.L_dot_S, self.L_tens_S]:
            if d:
                assert any(np.any(v) for v in d.values())

        self._validate_input(n_atoms)

    def validate_for_magnons(self, n_atoms: int) -> None:
        """
        Validates that the material properties are suitable for magnons. Namely, at least one of lambda_S and lambda_L must be defined.

        Args:
            n_atoms (int): Number of atoms in the material.

        Raises:
            AssertionError: If any of the required material properties for magnons are missing or have incorrect dimensions.

        """
        assert any([np.any(self.lambda_S), np.any(self.lambda_L)])
        # TODO: not nice to have so many return values
        self._validate_input(n_atoms)

    def _validate_input(self, n_atoms: int) -> None:
        """
        Validates the input to the MaterialProperties class has the correct shape and dictionary keys, etc. If anything is missing, it is filled in with reasonable default values.

        Args:
            n_atoms (int): Number of atoms in the material.

        Raises:
            AssertionError: If any of the material properties have incorrect dimensions or missing values.

        """

        psi = ["e", "p", "n"]
        self.N = self.N or {p: np.zeros(n_atoms) for p in psi}
        self.L_dot_S = self.L_dot_S or {p: np.zeros(n_atoms) for p in psi}
        self.S = self.S or {p: np.zeros((n_atoms, 3)) for p in psi}
        self.L = self.L or {p: np.zeros((n_atoms, 3)) for p in psi}
        self.L_tens_S = self.L_tens_S or {p: np.zeros((n_atoms, 3, 3)) for p in psi}
        self.lambda_L = np.zeros(n_atoms) if self.lambda_L is None else self.lambda_L
        self.lambda_S = np.zeros(n_atoms) if self.lambda_S is None else self.lambda_S

        # Populate default masses if necessary
        self.m_psi = self.m_psi or {
            "e": const.m_e,
            "p": const.m_p,
            "n": const.m_n,
        }

        # TODO: need proper exceptions
        # Validate that each dict has "e", "p" and "n" keys
        for d in [
            self.N,
            self.S,
            self.L,
            self.L_dot_S,
            self.L_tens_S,
            self.m_psi,
        ]:
            assert set(d.keys()) == set(psi)
        # Validate that the N and L_S dicts, each key is array of length num_atoms
        for d in [self.N, self.L_dot_S]:
            assert all(len(v) == n_atoms for v in d.values())
        # Assert that the S and L dicts, each key is array of length 3*num_atoms
        for d in [self.S, self.L]:
            assert all(v.shape == (n_atoms, 3) for v in d.values())
        assert all(v.shape == (n_atoms, 3, 3) for v in self.L_tens_S.values())
        # Assert that the lambda_S and lambda_L are arrays of length num_atoms
        assert len(self.lambda_S) == n_atoms
        assert len(self.lambda_L) == n_atoms


class Material(ABC):
    """
    Represents a generic material with its structural and atomic properties.

    Attributes:
        name (str): The name of the material.
        properties (MaterialProperties): The properties of the material.
        real_frac_to_cart (ndarray): The transformation matrix from fractional to Cartesian coordinates (units 1/eV), in real space.
        real_cart_to_frac (ndarray): The transformation matrix from Cartesian (units 1/eV) to fractional coordinates, in real space.
        recip_frac_to_cart (ndarray): The transformation matrix from fractional to Cartesian coordinates (units eV), in k-space.
        recip_cart_to_frac (ndarray): The transformation matrix from Cartesian (units eV) to fractional coordinates, in k-space.
        m_atoms (ArrayLike): an array of atomic masses, in eV.
        m_cell (ndarray): The total mass of the atoms in the material, in eV.
        xj (ndarray): The Cartesian coordinates (units 1/eV) of the atoms in the material.
        structure (Structure): the crystal structure `pymatgen` `Structure` object.
        n_atoms (int): The number of atoms in the material.
    """

    def __init__(
        self,
        name: str,
        properties: MaterialParameters,
        structure: Structure,
        m_atoms: ArrayLike,
    ):
        """
        Constructor for a generic Material object

        Args:
            name (str): The name of the material.
            properties (MaterialProperties): The properties of the material.
            structure (Structure): The structure of the material.
            m_atoms (ArrayLike): atomic masses in eV.
        """
        # Material properties
        self.name = name
        self.properties = properties

        # Define transformation matrices
        self.real_frac_to_cart = structure.lattice.matrix.T
        self.real_cart_to_frac = LA.inv(self.real_frac_to_cart)
        self.recip_frac_to_cart = structure.lattice.reciprocal_lattice.matrix.T
        self.recip_cart_to_frac = LA.inv(self.recip_frac_to_cart)

        # Atomic and structural properties
        self.m_atoms = m_atoms
        self.m_cell = np.sum(m_atoms)
        self.xj = structure.cart_coords
        self.structure = structure
        self.n_atoms = len(structure.species)

        # Internal variables
        self._max_dE = None
        self._q_cut = None

    @abstractmethod
    def get_eig(
        self, grid: SphericalGrid | MonkhorstPackGrid, with_eigenvectors: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Abstract method for computing eigenvalues and eigenvectors
        """
        pass

    @property
    @abstractmethod
    def max_dE(self) -> float:
        """
        Abstract method for estimating the maximum energy deposition
        """
        pass

    @property
    def q_cut(self) -> float:
        """
        Abstract method for estimating a cutoff for the momentum transfer
        """
        pass


class PhononMaterial(Material):
    """
    A class for materials with phonons.

    Attributes:
        phonopy_file (Phonopy): The Phonopy object for the material's phonons
        n_modes (int): The number of phonon modes in the material.
        born (np.ndarray): The born effective charges
        epsilon (np.ndarray): The dielectric tensor
    """

    def __init__(
        self, name: str, properties: MaterialParameters, phonopy_yaml_path: str
    ):
        """
        Constructor for PhononMaterial objects.

        Args:
            name (str): The name of the material.
            properties (MaterialProperties): The properties of the material.
            phonopy_yaml_path (str): The path to the Phonopy YAML file.

        """
        # TODO: Need a check for when phonopy_yaml does not have NAC
        phonopy_file = phonopy.load(phonopy_yaml=phonopy_yaml_path, is_nac=True)
        # TODO: should be a dict that has the correct factor for all codes
        length_factor = const.bohr_to_Ang if phonopy_file.calculator == "qe" else 1.0
        self.phonopy_file = phonopy_file
        n_atoms = phonopy_file.primitive.get_number_of_atoms()
        self.n_modes = 3 * n_atoms

        properties.validate_for_phonons(n_atoms)

        m_atoms = phonopy_file.primitive.masses * const.amu_to_eV

        # NAC parameters (born effective charges and dielectric tensor)
        self.born = np.array(
            phonopy_file.nac_params.get("born", np.zeros((n_atoms, 3, 3)))
        )
        self.epsilon = np.array(
            phonopy_file.nac_params.get("dielectric", np.identity(3))
        )

        # Create a Structure object
        # At some point should make careful assessment of primitive vs unit_cell
        # PhonoDark uses primitive, but what about when it's different from unit_cell?
        positions = phonopy_file.primitive.scaled_positions
        lattice = (
            np.array(phonopy_file.primitive.cell) * const.Ang_to_inveV * length_factor
        )
        species = phonopy_file.primitive.symbols

        structure = Structure(lattice, species, positions)

        super().__init__(name, properties, structure, m_atoms)

    def get_eig(
        self, grid: SphericalGrid, with_eigenvectors: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the phonon frequencies and eigenvectors for the given k-points.

        Args:
            grid (SphericalGrid): The grid object with the k-points.
            with_eigenvectors (bool, optional): Whether to compute the eigenvectors. Defaults to True.

        Returns:
            A tuple containing the phonon frequencies and eigenvectors.

                * The phonon frequencies are represented as a numpy array of shape (n_k,n_modes)

                * The eigenvectors are represented as a numpy array of shape (n_k, n_modes, n_atoms, 3)

                where n_k is the number of k-points, n_modes is the number of modes,
                n_atoms is the number of atoms, and the last index is
                for the x, y, z components of the eigenvectors.
        """
        k_points = grid.k_frac
        self.phonopy_file.run_qpoints(k_points, with_eigenvectors=with_eigenvectors)

        mesh_dict = self.phonopy_file.get_qpoints_dict()
        eigenvectors_pre = mesh_dict.get("eigenvectors", None)

        # Convert from THz (phonopy default for any calculator) to eV
        omega = const.THz_to_eV * mesh_dict["frequencies"]

        eigenvectors = np.zeros(
            (len(k_points), self.n_modes, self.n_atoms, 3), dtype=complex
        )
        # Need to reshape the eigenvectors from (n_k, n_modes, n_modes)
        # to (n_k, n_modes, n_atoms, 3)
        if with_eigenvectors:
            # TODO: Should rewrite this with a reshape...
            for q in range(len(k_points)):
                for nu in range(self.n_modes):
                    eigenvectors[q, nu] = np.array_split(
                        eigenvectors_pre[q].T[nu], self.n_atoms
                    )

        return omega, eigenvectors

    @property
    def max_dE(self) -> float:
        """
        Returns omega_ph_max = max(omega_ph) if there are optical modes, otherwise returns the average over the entire Brillouin zone. The quantities are obviously not the same but should be the same order. See theoretical framework paper, paragraph in middle of page 24 (of published version).

        TODO: clarify this

        Returns:
            float: the maximum energy deposition

        """
        if self._max_dE is None:
            if self.phonopy_file.primitive.get_number_of_atoms() == 1:
                self.phonopy_file.run_mesh([20, 20, 20])
                mesh_dict = self.phonopy_file.get_mesh_dict()
                weights = mesh_dict["weights"]
                omega = const.THz_to_eV * mesh_dict["frequencies"]
                # 2 is arbitrary
                self._max_dE = 2 * np.mean(np.average(omega, axis=0, weights=weights))
            else:
                self.phonopy_file.run_qpoints([[0, 0, 0]])
                omega_gamma = (
                    self.phonopy_file.get_qpoints_dict()["frequencies"]
                    * const.THz_to_eV
                )
                # 1.5 is arbitrary
                self._max_dE = 1.5 * np.amax(omega_gamma)
        return self._max_dE

    @property
    def q_cut(self) -> float:
        """
        The Debye-Waller factor suppresses the rate at larger q beyond
        q ~ np.sqrt(m_atom * omega_ph). This method calculates an estimate
        for the cutoff value of q.

        Returns:
            float: The cutoff value of q.

        """
        if self._q_cut is None:
            self._q_cut = 10.0 * np.sqrt(np.amax(self.m_atoms) * self.max_dE / 1.5)
        return self._q_cut

    def get_W_tensor(self, grid: MonkhorstPackGrid) -> np.ndarray:
        r"""
        Computes the W tensor for the given Monkhorst-Pack grid. The W tensor for atom $j$ is given by:
        $$
        \mathbf{W}_j = \frac{\Omega}{4 m_j} \sum_\nu \int_\text{1BZ} \frac{d^3k}{(2\pi)^3} \frac{\epsilon_{\nu j \bm{k}} \otimes \epsilon_{\nu j \bm{k}}^*}{\omega_{\nu \bm{k}}}
        $$

        The Debye-Waller factor can be computed from the W tensor as:
        $$
        W_j(\bm{q}) = \bm{q} \cdot (\mathbf{W}_j \bm{q})
        $$

        Args:
            grid (MonkhorstPackGrid): The Monkhorst-Pack grid.

        Returns:
            np.ndarray: The W tensor.

        """
        omega, epsilon = self.get_eig(grid)
        # epsilon is (n_k, n_modes, n_atoms, 3)
        eps_tensor = np.einsum("...i,...j->...ij", epsilon, np.conj(epsilon))

        # Sum over all modes and divide by the frequency
        W = (
            1
            / (4 * self.m_atoms[None, :, None, None])
            * np.sum(eps_tensor / omega[..., None, None, None], axis=1)
        )
        # Integrate over the BZ
        return np.sum(W * grid.weights[:, None, None, None], axis=0) / np.sum(
            grid.weights
        )


class MagnonMaterial(Material):
    """
    A class for materials with magnons

    Attributes:
        hamiltonian (SpinHamiltonian): The spin Hamiltonian of the material.
        n_modes (int): The number of magnon modes.
        dispersion (MagnonDispersion): The magnon dispersion.
    """

    def __init__(
        self,
        name: str,
        properties: MaterialParameters,
        hamiltonian: SpinHamiltonian,
        m_cell: float,
        nodmi: bool = False,
        noaniso: bool = False,
    ):
        """
        Constructor for a magnon material

        In the current implementation, the hamiltonian only
        contains the magnetic atoms and their interactions.
        So m_cell needs to be specified separately

        Args:
            name: The name of the material.
            properties: The properties of the material.
            hamiltonian: The spin Hamiltonian
            m_cell: the total mass of all ions in the cell
            nodmi: Whether to include DM interactions.
            noaniso: Whether to include anisotropic exchange.
        """
        # Ensure the hamiltonian is in the correct units
        # hamiltonian.cell *= const.Ang_to_inveV
        # In the future we should have a check that it comes in in units of A
        # And convert it here
        self.hamiltonian = hamiltonian
        n_atoms = len(hamiltonian.magnetic_atoms)
        self.n_modes = n_atoms
        self.dispersion = MagnonDispersion(
            hamiltonian, phase_convention="tanner", nodmi=nodmi, noaniso=noaniso
        )

        n_atoms = len(hamiltonian.magnetic_atoms)  # Number of magnetic atoms
        properties.validate_for_magnons(n_atoms)
        # Atom positions in cartesian coordinates (units of 1/eV)
        self.xj = np.array(
            [
                hamiltonian.get_atom_coordinates(atom, relative=False)
                for atom in hamiltonian.magnetic_atoms
            ]
        )
        # Spins
        self.Sj = np.array([atom.spin for atom in hamiltonian.magnetic_atoms])
        # The vectors for rotating to local coordiante system
        # TODO: make this an internal variable
        self.rj = self.dispersion.u

        positions = np.array([a.position for a in hamiltonian.magnetic_atoms])
        lattice = hamiltonian.cell
        species = [a.type for a in hamiltonian.magnetic_atoms]
        structure = Structure(lattice, species, positions)

        m_atoms = [m_cell / n_atoms] * n_atoms
        super().__init__(name, properties, structure, m_atoms)

    def get_eig(self, grid: SphericalGrid) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the magnon eigenvalues and eigenvectors (polarization vectors) for a list of k-points.

        Args:
            k (ArrayLike): The k-point in fractional coordinates.

        Returns:

        """
        omegas = np.zeros((grid.nq, self.n_modes))
        epsilons = np.zeros((grid.nq, self.n_modes, self.n_atoms, 3), dtype=complex)

        # TODO: This is a very slow implementation
        for ik, k in enumerate(grid.k_cart):
            omegas[ik, ...], epsilons[ik, ...] = self._get_eig_k(k)

        return omegas, epsilons

    def _get_eig_k(self, k: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the eigenvalues and magnon polarization vectors for a given k-point

        Args:
            k (ArrayLike): Single k-point, cartesian coordinates (units of eV).
            G (ArrayLike): Single G-vector, cartesian coordinates (units of eV).

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the eigenvalues (omega_nu) and eigenvectors (epsilon_nu_j).
                - omega_nu: (N,) an array of real numbers representing the eigenvalues in eV.
                - epsilon_nu_j: (N,N,3) array of complex numbers representing the eigenvectors (magnon polarization vectors), indexed as mode, atom, xyz
                N is the number of magnon modes.

        """

        # See Tanner's Disseration and RadTools doc for explanation of the 1/2
        omega_nu, Tk = self._get_omega_T(self.dispersion.h(k) / 2)
        N = self.n_atoms
        Uk_conj = np.conjugate(Tk[:N, :N])  # This is U_{j,nu,k})*
        V_minusk = np.conjugate(Tk[N:, :N])  # This is ((V_{j,nu,-k})*)*

        # Need to sort out the nu vs j index situation...
        epsilon_nu_j = (
            V_minusk[..., None] * (self.rj[:, None, :]).conj()
            + Uk_conj[..., None] * self.rj[:, None, :]
        )

        return omega_nu, epsilon_nu_j  # (n,) and (n,3) array of complex numbers

    def _get_omega_T(self, D) -> Tuple[np.ndarray, np.ndarray]:
        """
        TODO: clarify this and add references

        Calculate the eigenvalues and eigenvectors of the grand dynamical matrix.
        N is the number of magnon modes. This uses colpa's Algorithm

        Parameters:
        - D: grand dynamical matrix (2N x 2N)

        Returns:
        - omega: eigenvalues of the grand dynamical matrix (N,)
        - T: transformation matrix between
        """
        N = self.n_atoms
        g = np.diag([1] * N + [-1] * N)

        # We want D = K^dagger K whereas numpy uses K K^dagger
        K = np.conjugate(np.linalg.cholesky(D)).T
        L, U = np.linalg.eig(K @ g @ np.conjugate(K).T)

        # Arrange so that the first N are positive
        # And last N are negatives of the first N
        # (see Eq. I4 in Tanner's Dissertation)
        sort_order = np.argsort(L)[::-1]
        # Note: Testing shows that the exact sort order (Tanner's Eq. I4) is not necessary
        # The SpinW (Lake and Toth) sort order works just fine
        sort_order = np.concatenate((sort_order[:N], sort_order[2 * N : N - 1 : -1]))
        U = U[:, sort_order]
        L = np.diag(L[sort_order])

        # Now E is 1/2*diag([omega_1, omega_2, ..., omega_N,
        #                    omega_1, omega_2, ..., omega_N])
        # See Eq. (I4) in Tanner's Dissertation
        E = g @ L
        omega = 2 * np.diag(np.real(E[:N]))
        T = np.linalg.inv(K) @ U @ np.sqrt(E)

        return omega, T

    @property
    def max_dE(self) -> float:
        """
        TODO: this needs improvement

        Returns the maximum dE possible for the material.
        For magnons, we estimate this as roughly 3 times the highest magnon frequency
        at the Brillouin zone (BZ) boundary. If there are no gapped modes at the gamma point,
        the maximum dE will be 0.

        Returns:
            float: The maximum dE value.

        Notes:
            This calculation should be an average over the Brillouin zone (BZ).

        """
        # TODO: this should be an average over the BZ
        if self._max_dE is None:
            k = [1 / 2, 0, 0]
            omega, _ = self._get_eig_k(self.recip_frac_to_cart @ k)
            self._max_dE = 3 * np.amax(omega)
        return self._max_dE

    @property
    def q_cut(self) -> float:
        """
        For magnons there is no `q_cut`, so we just set this to a very large number.

        Returns:
            q_cut: a very large number.
        """
        return 1e10
