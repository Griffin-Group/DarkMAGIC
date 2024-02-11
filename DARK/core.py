import numpy as np
from numpy import linalg as LA
from pymatgen.core.structure import Structure
import phonopy

import DARK.constants as const

class Material():
    def __init__(self, name, structure, num_e, num_p, num_n, atomic_masses):
        self.name = name
        self.structure = structure
        self.num_e = num_e
        self.num_p = num_p
        self.num_n = num_n
        self.n_atoms = structure.num_sites

        # Define transformation matrices
        # Transpose is to keep up with original PhonoDark convention
        self.real_frac_to_cart = structure.lattice.matrix.T
        self.real_cart_to_frac = LA.inv(self.real_frac_to_cart)
        self.recip_frac_to_cart = structure.lattice.reciprocal_lattice.matrix.T
        self.recip_cart_to_frac = LA.inv(self.recip_frac_to_cart)

class PhononMaterial(Material):
    def __init__(self, name, phonopy_yaml_path):
        # TODO: Need a check for when phonopy_yaml does not have NAC
        phonopy_file = phonopy.load(phonopy_yaml=phonopy_yaml_path, is_nac=True)
        self.phonopy_file = phonopy_file
        n_atoms = phonopy_file.primitive.get_number_of_atoms()
        self.n_modes = 3 * n_atoms

        A = phonopy_file.primitive.get_masses() # Mass numbers
        Z = phonopy_file.primitive.get_atomic_numbers() # Atomic numbers  
        num_e = Z
        num_p = Z
        num_n = A-Z
        atomic_masses = A * const.amu_to_eV

        # NAC parameters (born effective charges and dielectric tensor)
        self.born = np.array(phonopy_file.nac_params.get('born', np.zeros((n_atoms, 3, 3))))
        self.epsilon = np.array(phonopy_file.nac_params.get('dielectric', np.identity(3)))
         
        # Create a Structure object
        # At some point should make careful assessment of primitive vs unit_cell
        # PhonoDark uses primitive, but what about when it's different from unit_cell?
        positions = phonopy_file.primitive.get_scaled_positions()
        lattice = np.array(phonopy_file.primitive.get_cell()) * const.Ang_to_inveV
        species = phonopy_file.primitive.get_chemical_symbols()

        structure = Structure(lattice, species, positions)

        super().__init__(name, structure, num_e, num_p, num_n, atomic_masses)

    def get_eig(self, mesh, with_eigenvectors=True):
        # run phonopy in mesh mode 
        self.phonopy_file.run_qpoints(mesh, with_eigenvectors=with_eigenvectors)


        mesh_dict = self.phonopy_file.get_qpoints_dict()

        eigenvectors_pre = mesh_dict.get('eigenvectors', None)
        # convert frequencies to correct units
        omega = const.THz_to_eV*mesh_dict['frequencies']

        n_k = len(mesh)

        # q, nu, i, alpha
        # Need to reshape the eigenvectors from (n_k, n_modes, n_modes) 
        # to (n_k, n_atoms, n_modes, 3)
        eigenvectors = np.zeros((len(mesh), self.n_modes, self.n_atoms, 3), dtype=complex)
        # Should rewrite this with a reshape...
        for q in range(n_k):
            for nu in range(self.n_modes):
                eigenvectors[q,nu] = np.array_split(
                        eigenvectors_pre[q].T[nu], self.n_atoms)

        return omega, eigenvectors
    
class MagnonMaterial(Material):
    def __init__(self):
        print("Not implemented yet!")

# TODO: c_dict and c_dict_form should prob just be merged?
class Model:
    def __init__(self, name, c_dict, c_dict_form, m_chi=None, times=None, Fmed_power=0, power_V=0, s_chi=0.5):
        """
        name: string
        m_chi: list of floats, DM masses (eV)
        times: list of floats, time of day for calculating earth velocity vector
        Fmed_power: float, negative power of q in the Fmed term
        power_V: float, power of q in the V term (for special mesh)
        s_chi float, spin of DM particle
        """
        self.name = name

        if m_chi is None:
            m_chi = np.logspace(3, 7, 50)
        self.m_chi = m_chi
        if times is None:
            times = [0]
        self.times = times
        self.Fmed_power = Fmed_power
        self.power_V = power_V
        self.s_chi = s_chi
