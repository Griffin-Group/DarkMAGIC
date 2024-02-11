import numpy as np
from numpy import linalg as LA
from pymatgen.core.structure import Structure
import phonopy
from radtools import MagnonDispersion

import DARK.constants as const


class Material:
    def __init__(self, name, structure, num_e, num_p, num_n, m_atoms):
        self.name = name
        self.structure = structure
        self.num_e = num_e
        self.num_p = num_p
        self.num_n = num_n
        self.n_atoms = len(structure.species)

        # Define transformation matrices
        # Transpose is to keep up with original PhonoDark convention
        self.real_frac_to_cart = structure.lattice.matrix.T
        self.real_cart_to_frac = LA.inv(self.real_frac_to_cart)
        self.recip_frac_to_cart = structure.lattice.reciprocal_lattice.matrix.T
        self.recip_cart_to_frac = LA.inv(self.recip_frac_to_cart)

        self.m_atoms = m_atoms
        self.m_cell = np.sum(m_atoms)
        self.xj = structure.cart_coords


class PhononMaterial(Material):
    def __init__(self, name, phonopy_yaml_path):
        # TODO: Need a check for when phonopy_yaml does not have NAC
        phonopy_file = phonopy.load(phonopy_yaml=phonopy_yaml_path, is_nac=True)
        self.phonopy_file = phonopy_file
        n_atoms = phonopy_file.primitive.get_number_of_atoms()
        self.n_modes = 3 * n_atoms

        A = phonopy_file.primitive.get_masses()  # Mass numbers
        Z = phonopy_file.primitive.get_atomic_numbers()  # Atomic numbers
        num_e = Z
        num_p = Z
        num_n = A - Z
        m_atoms = A * const.amu_to_eV

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

        super().__init__(name, structure, num_e, num_p, num_n, m_atoms)

    def get_eig(self, k_points, with_eigenvectors=True):
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
        eigenvectors = np.zeros(
            (len(k_points), self.n_modes, self.n_atoms, 3), dtype=complex
        )
        # TODO: Should rewrite this with a reshape...
        for q in range(n_k):
            for nu in range(self.n_modes):
                eigenvectors[q, nu] = np.array_split(
                    eigenvectors_pre[q].T[nu], self.n_atoms
                )

        return omega, eigenvectors


class MagnonMaterial(Material):
    def __init__(self, name, hamiltonian, m_cell):
        """
        In the current implementation, the hamiltonian only
        contains the magnetic atoms and their interactions.
        So m_cell needs to be specified separately
        """
        self.hamiltonian = hamiltonian
        n_atoms = len(hamiltonian.magnetic_atoms)
        self.n_modes = n_atoms
        self.dispersion = MagnonDispersion(hamiltonian, phase_convention="tanner")

        n_atoms = len(hamiltonian.magnetic_atoms)  # Number of magnetic atoms
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
        lattice = hamiltonian.cell * const.Ang_to_inveV
        species = [a.type for a in hamiltonian.magnetic_atoms]
        structure = Structure(lattice, species, positions)

        m_atoms = [m_cell / n_atoms] * n_atoms
        super().__init__(name, structure, None, None, None, m_atoms)

    def get_eig(self, k, G):
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


# TODO: c_dict and c_dict_form should prob just be merged?
class Model:
    def __init__(self, name, c_dict, c_dict_form, Fmed_power=0, power_V=0, s_chi=0.5):
        """
        name: string
        Fmed_power: float, negative power of q in the Fmed term
        power_V: float, power of q in the V term (for special mesh)
        s_chi float, spin of DM particle
        """
        self.name = name

        self.Fmed_power = Fmed_power
        self.power_V = power_V
        self.s_chi = s_chi


class Numerics:
    def __init__(
        self,
        N_abc: np.array | list = [50, 25, 25],
        power_abc: np.array | list = [2, 1, 1],
        n_DW_xyz: np.array | list = [20, 20, 20],
        bin_width: float = 1e-3,
        q_cut: bool = True,
        special_mesh: bool = True,
    ):
        self.N_abc = np.array(N_abc)
