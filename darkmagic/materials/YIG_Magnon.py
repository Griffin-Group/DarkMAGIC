from copy import deepcopy

import numpy as np
from pymatgen.core.structure import Structure
from radtools import Atom, Crystal, ExchangeParameter, Lattice, SpinHamiltonian

import darkmagic.constants as const
from darkmagic.material import MagnonMaterial, MaterialProperties


def get_material():
    def prepare_YIG_structure(filename):
        struct = Structure.from_file(filename)
        struct.remove_species(["Y", "O"])
        for i in range(12):
            struct.sites[i].species = "Fe1"
        for j in range(12, 20):
            struct.sites[j].species = "Fe2"
        return struct

    def get_YIG_hamiltonian(filename):
        struct = prepare_YIG_structure(filename)

        def site_type(i):
            return "d" if i < 12 else "a"

        # See blue notebook notes page 31 regarding why this is frac_coords
        # Even though documentation says it should be cartesian ("absolute")
        atoms = [
            Atom(f"Fe_{site_type(i)}", s.frac_coords, index=i + 1, spin=5 / 2)
            for i, s in enumerate(struct.sites)
        ]

        for atom in atoms:
            atom.spin_direction = [0, 0, 1] if atom.index < 13 else [0, 0, -1]
        # Convert everything to eV and inverse eV
        scaled_struct = deepcopy(struct)
        scaled_struct.scale_lattice(struct.volume * (const.Ang_to_inveV) ** 3)

        # Having to pass standardize = False every round because of inheritance is a bit absurd
        lattice = Lattice(
            np.round(scaled_struct.lattice.matrix, decimals=7), standardize=False
        )
        crystal = Crystal(lattice, atoms, standardize=False)
        hamiltonian = SpinHamiltonian(crystal, standardize=False)

        kelvin_to_eV = 8.617333262145 * 1e-5

        # From that paper cited by Tanner
        Jad = ExchangeParameter(iso=-40 * kelvin_to_eV)
        Jdd = ExchangeParameter(iso=-13.4 * kelvin_to_eV)
        Jaa = ExchangeParameter(iso=-3.8 * kelvin_to_eV)
        J = {"aa": Jaa, "dd": Jdd, "ad": Jad, "da": Jad}

        hamiltonian.double_counting = False
        hamiltonian.spin_normalized = False  # I don't think it's normalized??
        hamiltonian.factor = -2

        # Add all pairs
        all_indices, all_distances, all_R = get_YIG_neighbors(struct)
        for current_index, (neighbor_indices, neighbor_R) in enumerate(
            zip(all_indices, all_R)
        ):
            for neighbor_index, R in zip(neighbor_indices, neighbor_R):
                parameter_type = site_type(current_index) + site_type(neighbor_index)
                hamiltonian.add_bond(
                    atoms[current_index],
                    atoms[neighbor_index],
                    tuple(R),
                    J[parameter_type],
                )

        return hamiltonian

    def get_YIG_neighbors(struct, neighbor_cutoff=5.5):
        """
        Returns the indices, distances, and R vectors of the nearest neighbors
        neighbor_cutoff: distance in Angstroms
        """
        indices = []
        distances = []
        R = []
        for current_site in struct.sites:
            neighbors = struct.get_neighbors(current_site, neighbor_cutoff)
            neighbor_indices = [n.index for n in neighbors]
            neighbor_distance = [n.distance(current_site) for n in neighbors]
            neighbor_R = [
                n.frac_coords - n.to_unit_cell().frac_coords for n in neighbors
            ]
            distance_sort = np.argsort(neighbor_distance)
            neighbor_indices = np.array(neighbor_indices)[distance_sort]
            neighbor_distance = np.array(neighbor_distance)[distance_sort]
            neighbor_R = np.array(neighbor_R)[distance_sort]
            indices.append(neighbor_indices)
            distances.append(neighbor_distance)
            R.append(neighbor_R)
            # Save results

        return indices, distances, R

    hamiltonian = get_YIG_hamiltonian("data/YIG.vasp")
    m_cell = 2749.367e9  # YIG mass, all ions
    n_atoms = len(hamiltonian.magnetic_atoms)

    # Temporary for testing
    N = {
        "e": np.random.rand(n_atoms),
        "p": np.random.rand(n_atoms),
        "n": np.zeros(n_atoms),
    }
    L = {
        "e": np.random.rand(n_atoms, 3),
        "p": np.ones((n_atoms, 3)),
        "n": np.zeros((n_atoms, 3)),
    }
    L_tens_S = {
        "e": np.random.rand(n_atoms, 3, 3),
        "p": np.ones((n_atoms, 3, 3)),
        "n": np.zeros((n_atoms, 3, 3)),
    }

    properties = MaterialProperties(
        N=N, L=L, L_tens_S=L_tens_S, lambda_S=np.ones(n_atoms)
    )
    return MagnonMaterial("YIG", properties, hamiltonian, m_cell)
