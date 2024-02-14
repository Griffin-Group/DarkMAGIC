import itertools

import numpy as np
from numpy.typing import ArrayLike
from numpy import linalg as LA
from pymatgen.core.structure import Structure
import phonopy
from radtools import SpinHamiltonian, MagnonDispersion


import DARK.constants as const


class MaterialProperties:
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

    def validate_for_phonons(self, n_atoms: int):
        """
        Validates that the material properties are suitable for phonons
        """
        assert any([self.N, self.S, self.L, self.L_dot_S, self.L_tens_S])
        self._validate_input(n_atoms)

    def validate_for_magnons(self, n_atoms: int):
        """
        Validates that the material properties are suitable for magnons
        """
        assert any([np.any(self.lambda_S), np.any(self.lambda_L)])
        # TODO: not nice to have so many return values
        self._validate_input(n_atoms)

    def _validate_input(self, n_atoms):
        """
        Validates the input to the MaterialProperties class
        """

        psi = ["e", "p", "n"]

        self.N = self.N or {k: np.zeros(n_atoms) for k in psi}
        self.L_dot_S = self.L_dot_S or {k: np.zeros(n_atoms) for k in psi}
        self.S = self.S or {k: np.zeros((n_atoms, 3)) for k in psi}
        self.L = self.L or {k: np.zeros((n_atoms, 3)) for k in psi}
        self.L_tens_S = self.L_tens_S or {k: np.zeros((n_atoms, 3, 3)) for k in psi}
        self.lambda_L = np.zeros(n_atoms) if self.lambda_L is None else self.lambda_L
        self.lambda_S = np.zeros(n_atoms) if self.lambda_S is None else self.lambda_S

        # Populate default masses if necessary
        self.m_psi = self.m_psi or {"e": const.m_e, "p": const.m_p, "n": const.m_n}

        # TODO: need proper exceptions
        # Validate that each dict has "e", "p" and "n" keys
        for d in [self.N, self.S, self.L, self.L_dot_S, self.L_tens_S, self.m_psi]:
            assert set(d.keys()) == set(psi)
        # Validate that the N and L_S dicts, each key is array of length num_atoms
        for d in [self.N, self.L_dot_S]:
            assert all(len(v) == n_atoms for v in d.values())
        # Assert that the S and L dicts, each key is array of length 3*num_atoms
        for d in [self.S, self.L]:
            assert all(v.shape == (n_atoms, 3) for v in d.values())
        assert all(v.shape == (n_atoms, 3, 3) for v in self.L_tens_S.values())


# TODO: make this an abstract class and define Phonon/MagnonMaterial as children
class Material:
    def __init__(
        self,
        name: str,
        properties: MaterialProperties,
        structure: Structure,
        m_atoms: ArrayLike,
    ):
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


class PhononMaterial(Material):
    # TODO: path should be "PathLike" (does that exist?)
    def __init__(
        self, name: str, properties: MaterialProperties, phonopy_yaml_path: str
    ):
        # TODO: Need a check for when phonopy_yaml does not have NAC
        phonopy_file = phonopy.load(phonopy_yaml=phonopy_yaml_path, is_nac=True)
        self.phonopy_file = phonopy_file
        n_atoms = phonopy_file.primitive.get_number_of_atoms()
        self.n_modes = 3 * n_atoms

        properties.validate_for_phonons(n_atoms)

        m_atoms = phonopy_file.primitive.get_masses() * const.amu_to_eV

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
        positions = phonopy_file.primitive.get_scaled_positions()
        lattice = np.array(phonopy_file.primitive.get_cell()) * const.Ang_to_inveV
        species = phonopy_file.primitive.get_chemical_symbols()

        structure = Structure(lattice, species, positions)

        super().__init__(name, properties, structure, m_atoms)

    def get_eig(self, k_points: ArrayLike, with_eigenvectors: bool = True):
        """
        k_points: numpy arrays of k-points, fractional coordinates
        """
        # run phonopy in mesh mode
        self.phonopy_file.run_qpoints(k_points, with_eigenvectors=with_eigenvectors)

        mesh_dict = self.phonopy_file.get_qpoints_dict()

        eigenvectors_pre = mesh_dict.get("eigenvectors", None)
        # convert frequencies to correct units
        omega = const.THz_to_eV * mesh_dict["frequencies"]

        n_k = len(k_points)

        # Need to reshape the eigenvectors from (n_k, n_modes, n_modes)
        # to (n_k, n_atoms, n_modes, 3)
        if with_eigenvectors:
            eigenvectors = np.zeros(
                (len(k_points), self.n_modes, self.n_atoms, 3), dtype=complex
            )
            # TODO: Should rewrite this with a reshape...
            for q in range(n_k):
                for nu in range(self.n_modes):
                    eigenvectors[q, nu] = np.array_split(
                        eigenvectors_pre[q].T[nu], self.n_atoms
                    )
        else:
            eigenvectors = None

        return omega, eigenvectors

    @property
    def max_dE(self):
        """
        Returns omega_ph_max = max(omega_ph) if there are optical modes, otherwise returns the average over the entire Brillouin zone. The quantities are obviously not the same but should be the same order. See theoretical framework paper, paragraph in middle of page 24 (of published version).
        """
        if self._max_dE is None:
            if self.phonopy_file.primitive.get_number_of_atoms() == 1:
                self.phonopy_file.run_mesh([20, 20, 20], with_eigenvectors=False)
                mesh_dict = self.phonopy_file.get_mesh_dict()
                weights = mesh_dict["weights"]
                omega = const.THz_to_eV * mesh_dict["frequencies"]
                self._max_dE = 2 * np.mean(np.average(omega, axis=0, weights=weights))
            else:
                omega, _ = self.get_eig([[0, 0, 0]], with_eigenvectors=False)
                self._max_dE = 1.5 * np.amax(omega)
        return self._max_dE

    @property
    def q_cut(self):
        """
        The Debye-Waller factor supresses the rate at larger q beyond
        q ~ np.sqrt(m_atom * omega_ph). This is an estimate for that
        cutoff.
        """
        if self._q_cut is None:
            self._q_cut = 10.0 * np.sqrt(np.amax(self.m_atoms) * self.max_dE)
        return self._q_cut


class MagnonMaterial(Material):
    def __init__(
        self,
        name: str,
        properties: MaterialProperties,
        hamiltonian: SpinHamiltonian,
        m_cell: float,
    ):
        """
        In the current implementation, the hamiltonian only
        contains the magnetic atoms and their interactions.
        So m_cell needs to be specified separately
        """
        # Ensure the hamiltonian is in the correct units
        # hamiltonian.cell *= const.Ang_to_inveV
        # In the future we should have a check that it comes in in units of A
        # And convert it here
        self.hamiltonian = hamiltonian
        n_atoms = len(hamiltonian.magnetic_atoms)
        self.n_modes = n_atoms
        self.dispersion = MagnonDispersion(hamiltonian, phase_convention="tanner")

        n_atoms = len(hamiltonian.magnetic_atoms)  # Number of magnetic atoms
        properties.validate_for_magnons(n_atoms)
        # Atom positions in cartesian coordinates (units of 1/eV)
        self.xj = np.array(
            [
                hamiltonian.get_atom_coordinates(atom, relative=False)
                for atom in hamiltonian.magnetic_atoms
            ]
        )
        # sqrt(Sj/2)
        self.sqrt_spins_2 = np.sqrt(
            np.array([atom.spin for atom in hamiltonian.magnetic_atoms]) / 2
        )
        # The vectors for rotating to local coordiante system
        self.rj = self.dispersion.u

        positions = np.array([a.position for a in hamiltonian.magnetic_atoms])
        lattice = hamiltonian.cell
        species = [a.type for a in hamiltonian.magnetic_atoms]
        structure = Structure(lattice, species, positions)

        m_atoms = [m_cell / n_atoms] * n_atoms
        super().__init__(name, properties, structure, m_atoms)

    def get_eig(self, k, G=[0, 0, 0]):
        """
        k: single k-point, cartesian coordinates (units of eV)
        G: single G-point, cartesian coordinates (units of eV)
        """
        # Calculate the prefactor
        prefactor = self.sqrt_spins_2 * np.exp(1j * np.dot(self.xj, G))

        # See Tanner's Disseration and RadTools doc for explanation of the 1/2
        N = self.n_atoms
        omega_nu_k, Tk = self._get_omega_T(self.dispersion.h(k) / 2)
        Uk_conj = np.conjugate(Tk[:N, :N])  # This is U_{j,nu,k})*
        V_minusk = np.conjugate(Tk[N:, :N])  # This is ((V_{j,nu,-k})*)*

        epsilon_nu_k_G = (prefactor[:, None] * V_minusk).T @ np.conjugate(self.rj) + (
            prefactor[:, None] * Uk_conj
        ).T @ self.rj

        return omega_nu_k, epsilon_nu_k_G  # n, and nx3 array of complex numbers

    def _get_omega_T(self, D):
        """
        D: grand dynamical matrix (2N x 2N)
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
        # sort_order = np.concatenate((sort_order[:N], sort_order[2 * N : N - 1 : -1]))
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
    def max_dE(self):
        """
        Returns the maximum dE possible for the material.
        For magnons, we estimate this as roughly 3 * the highest magnon frequency
        at the BZ boundary. (At gamma point will be 0 if there are no gapped modes).
        """
        # TODO: this should be an average over the BZ
        if self._max_dE is None:
            k = [1 / 2, 0, 0]
            omega, _ = self.get_eig(self.recip_frac_to_cart @ k)
            self._max_dE = 3 * np.amax(omega)
        return self._max_dE

    @property
    def q_cut(self):
        """
        For magnons there is no q_cut so we set this to a very large number.
        """
        return 1e10
