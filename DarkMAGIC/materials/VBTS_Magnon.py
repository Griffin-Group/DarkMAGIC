from copy import deepcopy

import numpy as np
from pymatgen.core.structure import Structure
from radtools import (
    Atom,
    Crystal,
    ExchangeParameter,
    Lattice,
    SpinHamiltonian,
)

import DarkMAGIC.constants as const
from DarkMAGIC.material import MagnonMaterial, MaterialProperties


def get_material():
    def prepare_MVBT_structure(filename):
        struct = Structure.from_file(filename)
        struct.remove_species(["Bi", "Te", "Se"])
        # struct.sites[0].species = struct.sites[0].species + "1"
        # struct.sites[1].species = struct.sites[0].species + "2"
        return struct

    def get_MVBT_neighbors(struct, neighbor_cutoff=23):
        indices = [{"inter": [], "intra": []} for i in range(struct.num_sites)]
        distances = [{"inter": [], "intra": []} for i in range(struct.num_sites)]
        R = [{"inter": [], "intra": []} for i in range(struct.num_sites)]
        z = [{"inter": [], "intra": []} for i in range(struct.num_sites)]
        for i, current_site in enumerate(struct.sites):
            neighbors = struct.get_neighbors(current_site, neighbor_cutoff)
            for n in neighbors:
                if n.index == i and n.coords[2] == current_site.coords[2]:
                    # Neighbor is same kind of site and at same z value
                    key = "intra"
                elif n.index != i and n.coords[2] != current_site.coords[2]:
                    # Neighbor is different kind of site and different z value
                    key = "inter"
                else:
                    # This is a neighbor of the same kind of site but from a different layer
                    # We neglect these
                    continue
                indices[i][key].append(n.index)
                R[i][key].append(n.image)
                distances[i][key].append(n.distance(current_site, [0, 0, 0]))
                z[i][key].append(n.coords[2])
            distance_sort = {k: np.argsort(distances[i][k]) for k in ["inter", "intra"]}
            for k in ["inter", "intra"]:
                indices[i][k] = np.array(indices[i][k])[distance_sort[k]]
                distances[i][k] = np.array(distances[i][k])[distance_sort[k]]
                R[i][k] = np.array(R[i][k])[distance_sort[k]]
                z[i][k] = np.array(z[i][k])[distance_sort[k]]
                neighbor_class = -1
                prev_d = 0
                new_indices, new_distances, new_R, new_z = [[]], [], [[]], [[]]
                for j, d, RR, zz in zip(
                    indices[i][k], distances[i][k], R[i][k], z[i][k]
                ):
                    if ~np.isclose(d, prev_d):
                        neighbor_class += 1
                        new_indices.append([])
                        new_distances.append(d)
                        new_R.append([])
                        new_z.append([])
                    new_indices[neighbor_class].append(j)
                    new_R[neighbor_class].append(RR)
                    new_z[neighbor_class].append(zz)
                    prev_d = d
                # Get rid of that empty list because this is an atrocious implementation
                indices[i][k] = new_indices[:-1]
                distances[i][k] = new_distances[:-1]
                R[i][k] = new_R[:-1]
                z[i][k] = new_z[:-1]

        return indices, distances, R, z

    def get_MVBT_hamiltonian(filename, J, spin_direction, spin):
        struct = prepare_MVBT_structure(filename)
        atoms = [
            Atom("V", s.frac_coords, index=i, spin=spin)
            for i, s in enumerate(struct.sites)
        ]
        for a in atoms:
            a.spin_direction = ((-1) ** a.index) * np.array(spin_direction)

        # Convert everything to eV and inverse eV
        scaled_struct = deepcopy(struct)
        scaled_struct.scale_lattice(struct.volume * (const.Ang_to_inveV) ** 3)
        lattice = Lattice(
            np.round(scaled_struct.lattice.matrix, decimals=7), standardize=False
        )
        crystal = Crystal(lattice, atoms, standardize=False)
        hamiltonian = SpinHamiltonian(crystal, standardize=False)

        J = {
            "inter": [ExchangeParameter(iso=JJ) for JJ in J["inter"]],
            "intra": [ExchangeParameter(iso=JJ) for JJ in J["intra"]],
        }

        # The paper is very unclear about their hamiltonian convention unfortunately
        hamiltonian.double_counting = True  # Really not sure about this...
        hamiltonian.spin_normalized = True  # I don't think it's normalized??
        hamiltonian.factor = -1 / 2  # Really not sure about this eitehr...

        # Add all pairs
        all_indices, _, all_R, _ = get_MVBT_neighbors(struct)
        for current_index, (neighbor_indices, neighbor_R) in enumerate(
            zip(all_indices, all_R)
        ):
            # Now this gives the index of the current site,
            # and dictionaries with the inter/intra layer neighbors
            for k in ["inter", "intra"]:
                for neighbor_class, (class_indices, class_R) in enumerate(
                    zip(neighbor_indices[k], neighbor_R[k])
                ):
                    # now each of these have neighbors in the same class (i.e., same distance away)
                    for neighbor_index, R in zip(class_indices, class_R):
                        hamiltonian.add_bond(
                            atoms[current_index],
                            atoms[neighbor_index],
                            tuple(R),
                            J[k][neighbor_class],
                        )
        return hamiltonian

    # hamiltonian = get_YIG_hamiltonian('symm_reduced_YIG.vasp')
    J = {
        "inter": np.array(
            [
                -0.00037656903765693300,
                0.00046025104602507400,
                -0.0030543933054393700,
                -0.0002928870292887300,
                0.0025523012552301100,
                0.0016317991631799000,
                -0.00020920502092053300,
                0.00029288702928867400,
                0.0012133891213388900,
                -0.002970711297071170,
                0.00004184100418407060,
                -0.0006276150627615370,
                -0.0033054393305439700,
                -0.0006276150627615370,
                0.00004184100418407060,
                0.00004184100418407060,
                -0.0005439330543933400,
                0.0001255230125522740,
            ]
        )
        * 1e-3,
        "intra": np.array(
            [
                2.4835425076352400,
                0.011087866108786600,
                0.061932773109243700,
                0.0024686192468619000,
                -0.0012970711297071400,
                0.0009623430962342820,
                -0.0005439330543933400,
                0.00046025104602507400,
                -0.0007949790794979370,
                0.0007112970711296780,
                -0.000041841004184133,
                -0.000041841004184133,
                0,
            ]
        )
        * 1e-3,
    }
    spin_direction = [1, 0, 0]
    spin = 3 / 2
    hamiltonian = get_MVBT_hamiltonian("data/VBTS.vasp", J, spin_direction, spin)
    m_cell = 2749.367e9  # YIG mass, all ions
    m_cell = 1643.2017317087846e9  # VBTS mass, magnetic cell
    n_atoms = len(hamiltonian.magnetic_atoms)
    properties = MaterialProperties(lambda_S=np.ones(n_atoms))
    return MagnonMaterial("VBTS", properties, hamiltonian, m_cell)
