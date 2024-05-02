import warnings
from typing import Tuple

import h5py
import numpy as np
import pkg_resources

from darkmagic.benchmark_models.utils import one
from darkmagic.parallel import ROOT_PROCESS

VERSION = my_version = pkg_resources.get_distribution("dark-magic").version


def read_h5(filename, format):
    """
    Reads data from an HDF5 file based on the specified format.

    Args:
        filename (str): The path to the HDF5 file to read.
        format (str): The format of the data to read. Should be either "phonodark" or "darkmagic".

    Returns:
        The data read from the HDF5 file based on the specified format.

    Raises:
        ValueError: If the provided format is not recognized.
    """

    if format == "phonodark":
        return read_phonodark(filename)
    elif format == "darkmagic":
        return read_darkmagic(filename)
    else:
        raise ValueError(f"Format {format} not recognized.")


def read_darkmagic(filename: str) -> Tuple[dict, dict, dict, dict]:
    """
    Reads data from an HDF5 file in the 'darkmagic' format.

    Args:
        filename (str): The path to the HDF5 file in DarkMAGIC format.

    Returns:
        Tuple containing the calculated values, numerics, model, and data in 'darkmagic' format.
    """
    with h5py.File(filename, "r") as f:
        calc = dict_from_h5group(f["calc"])
        numerics = dict_from_h5group(f["numerics"])
        model = dict_from_h5group(f["model"])
        data = dict_from_h5group(f["data"])
    return calc, numerics, model, data


def read_phonodark(filename: str) -> Tuple[dict, dict, dict, dict]:
    """
    Reads data from an HDF5 file in the 'phonodark' format and converts it to the 'darkmagic' format.

    Args:
        filename (str): The path to the HDF5 file in 'phonodark' format.

    Returns:
        Tuple containing the calculated values, numerics, model, and data converted to 'DarkMAGIC' format.
    """

    def get_phonodark_rate(group):
        return np.array(
            [
                [data[group][str(t)][str(m)] for m in range(len(masses))]
                for t in range(len(times))
            ]
        )

    with h5py.File(filename, "r") as f:
        particle_physics = dict_from_h5group(f["particle_physics"])
        numerics = dict_from_h5group(f["numerics"])
        data = dict_from_h5group(f["data"])
        times = particle_physics["times"]
        masses = particle_physics["dm_properties"]["mass_list"]

        data = {
            "diff_rate": get_phonodark_rate("diff_rate"),
            "binned_rate": get_phonodark_rate("binned_rate"),
            "total_rate": get_phonodark_rate("rate")[0].T,
        }

        calc, numerics, model = phonodark_to_darkmagic(
            numerics, particle_physics, times, masses
        )

    return calc, numerics, model, data


def write_h5(
    out_filename,
    material,
    model,
    numerics,
    masses,
    times,
    v_e,
    all_total_rate_list,
    all_diff_rate_list,
    all_binned_rate_list,
    rank,
    comm,
    parallel=False,
    format="darkmagic",
):
    # Write to file
    # TODO: is there a more succinct way to do this?
    # TODO: this should be cleaned up to not depend on MPI stuff and that low
    # level stuff can go elsewhere?
    if rank == ROOT_PROCESS:
        print(f"Writing to file {out_filename}{' in parallel' if parallel else ''}...")
    writer = write_phonodark if format == "phonodark" else write_darkmagic
    if not parallel and rank == ROOT_PROCESS:
        writer(
            out_filename,
            material,
            model,
            numerics,
            masses,
            times,
            v_e,
            all_total_rate_list,
            all_diff_rate_list,
            all_binned_rate_list,
            comm=None,
        )
    else:
        writer(
            out_filename,
            material,
            model,
            numerics,
            masses,
            times,
            v_e,
            all_total_rate_list,
            all_diff_rate_list,
            all_binned_rate_list,
            comm=comm,
        )


def write_darkmagic(
    out_file,
    material,
    model,
    numerics,
    m_chi,
    time,
    v_e,
    total_rate,
    diff_rate,
    binned_rate,
    comm=None,
):
    """
    Write data to hdf5 file in DarkMAGIC format.

    Args:
        out_file: Path to the output HDF5 file.
        material: Material object.
        model: Model object.
        numerics: Numerics object.
        m_chi: List of masses for the jobs.
        time: List of times for the jobs.
        all_total_rate_list: List of total rate data.
        all_diff_rate_list: List of differential rate data.
        all_binned_rate_list: List of binned rate data.
        comm: MPI communicator for parallel writing (default is None).

    Returns:
        None
    """

    # Get appropriate context manager for serial/parallel
    if comm is None:
        cm = h5py.File(out_file, "w")
    else:
        cm = h5py.File(out_file, "w", driver="mpio", comm=comm)

    with cm as out_f:
        h5group_to_dict(out_f, "numerics", numerics.to_dict())
        h5group_to_dict(out_f, "model", model.to_dict(serializable=True))
        calc = {
            "calc_type": "scattering",
            "m_chi": m_chi,
            "time": time,
            "v_e": v_e,
        }
        # Create groups/datasets and write out input parameters
        h5group_to_dict(out_f, "calc", calc)
        out_f.create_dataset("version", data=np.array([VERSION], dtype="S"))

        nt, nm = len(time), len(m_chi)
        num_bins = diff_rate.shape[-1]
        num_modes = binned_rate.shape[-1]

        out_f.create_dataset("data/diff_rate", shape=(nt, nm, num_bins), dtype="f8")
        out_f.create_dataset("data/binned_rate", shape=(nt, nm, num_modes), dtype="f8")
        out_f.create_dataset("data/total_rate", shape=(nt, nm), dtype="f8")

        out_f["data"]["diff_rate"][...] = diff_rate
        out_f["data"]["binned_rate"][...] = binned_rate
        out_f["data"]["total_rate"][...] = total_rate


def write_phonodark(
    out_file,
    material,
    model,
    numerics,
    masses,
    times,
    v_e,
    all_total_rate_list,
    all_diff_rate_list,
    all_binned_rate_list,
    comm=None,
):
    """

    Write data to hdf5 file in PhonoDark format

    all_*_rate_list has a complicated format.
    For starters, it has as many elements as there are jobs. Each element is a list of the
    (m_chi, time) pairs that have been computed. Each of those elements is a two element array.
    The first element is a tuple of the mass and time index. The second element is either a scalar
    (total) or an array (binned and diff). For binned and diff, the dimensions of the array are
    num_bins and num_modes respectively.

    So for example, if we have 4 masses and 2 times, distributed over 4 jobs, then diff_rate list will look like this:
    [
        # This is the job 1 array, which does mass index 0 at time indices 0,1
        [ [ [0, 0], [num_bins elements]], [[0, 1], [num_bins_elements], ...], # job 1
        # This is the job 1 array, which does mass index 1 at time indices 0,1
        [ [ [1, 0], [num_bins elements]], [[1, 1], [num_bins_elements], ...], # job 2
        ...
    ]
    This format is temporary, just for backwards compatibility with PhonoDark.

    Args:
      out_file: Path to the output HDF5 file.
      material: Material object.
      model: Model object.
      numerics: Numerics object.
      masses: List of masses for the jobs.
      times: List of times for the jobs.
      all_total_rate_list: List of total rate data.
      all_diff_rate_list: List of differential rate data.
      all_binned_rate_list: List of binned rate data.
      comm: MPI communicator for parallel writing (default is None).

    Returns:
      None
    """

    def get_dicts(model, numerics, masses, times):
        physics_parameters = {
            "threshold": numerics._threshold,
            "times": times,
            "Fmed_power": model.Fmed_power,
            "power_V": model.power_V,
            "special_model": False,
            "model_name": "mdm",
        }
        dm_properties_dict = {
            "spin": model.S_chi,
            "mass_list": masses,
        }
        coeff = model.coeff_prefactor
        numerics_parameters = {
            "n_a": numerics.N_grid[0],
            "n_b": numerics.N_grid[1],
            "n_c": numerics.N_grid[2],
            "power_a": numerics._power_abc[0],
            "power_b": numerics._power_abc[1],
            "power_c": numerics._power_abc[2],
            "n_DW_x": numerics.N_DWF_grid[0],
            "n_DW_y": numerics.N_DWF_grid[1],
            "n_DW_z": numerics.N_DWF_grid[2],
            "energy_bin_width": numerics.bin_width,
            "q_cut": numerics.use_q_cut,
            "special_mesh": numerics._use_special_mesh,
        }
        return physics_parameters, dm_properties_dict, coeff, numerics_parameters

    physics_parameters, dm_properties_dict, c_dict, numerics_parameters = get_dicts(
        model, numerics, masses, times
    )

    # Get appropriate context manager for serial/parallel
    cm = _get_context_manager(out_file, comm)

    with cm as h5f:
        # Create groups/datasets and write out input parameters
        h5group_to_dict(h5f, "numerics", numerics_parameters)
        h5group_to_dict(h5f, "particle_physics", physics_parameters)
        h5f.create_dataset("version", data=np.array(["0.0.1"], dtype="S"))
        h5group_to_dict(h5f, "particle_physics/dm_properties", dm_properties_dict)
        h5group_to_dict(h5f, "particle_physics/c_coeffs", c_dict)

        if isinstance(all_diff_rate_list, list):
            # Old implementation friendly
            num_bins = len(all_diff_rate_list[0][0][1])
            num_modes = len(all_binned_rate_list[0][0][1])
        else:
            num_bins = all_diff_rate_list.shape[-1]
            num_modes = all_binned_rate_list.shape[-1]

        # In parallel all datasets need to be created by all ranks
        for i in range(len(physics_parameters["times"])):
            for j in range(len(dm_properties_dict["mass_list"])):
                h5f.create_dataset(f"data/rate/{i}/{j}", shape=(1,), dtype="f8")
                h5f.create_dataset(
                    f"data/diff_rate/{i}/{j}", shape=(num_bins,), dtype="f8"
                )
                h5f.create_dataset(
                    f"data/binned_rate/{i}/{j}", shape=(num_modes,), dtype="f8"
                )
                # This will work for the new implementation where
                # the data is arrays indexed as (time, mass), but not
                # in parallel. See parallel implementation below.
                h5f[f"data/rate/{i}/{j}"][...] = all_total_rate_list[i, j]
                h5f[f"data/diff_rate/{i}/{j}"][...] = all_diff_rate_list[i, j]
                h5f[f"data/binned_rate/{i}/{j}"][...] = all_binned_rate_list[i, j]

            # Parallel friendly implementation
            # Write out rates to the file
            # To accomodate for serial, can flatten all_total_rate_list
            # for tr_list, br_list, dr_list in zip(
            #     all_total_rate_list, all_binned_rate_list, all_diff_rate_list
            # ):
            #     for t, b, d in zip(tr_list, br_list, dr_list):
            #         # The first element of t, b, and d is a tuple of the mass and time index
            #         # The second element is either a scalar (total) or an array (binned and diff)
            #         # For binned and diff, the dimensions of the array are num_bins
            #         # and num_modes respectively

            #         # mass_index, time_index
            #         j, i = map(str, map(int, t[0]))
            #         # print(f'Writing data/rate/{i}/{j} = {t[1]} to {out_f["data"]["rate"][i][j]}')
            #         h5f["data"]["rate"][i][j][...] = t[1]
            #         h5f["data"]["binned_rate"][i][j][...] = b[1]
            #         h5f["data"]["diff_rate"][i][j][...] = d[1]


def _get_context_manager(out_file, comm):
    if comm is None:
        cm = h5py.File(out_file, "w")
    else:
        cm = h5py.File(out_file, "w", driver="mpio", comm=comm)
    return cm


def dict_from_h5group(group: h5py.Group):
    """
    Recurses through an h5py group and creates a dictionary with the same structure.
    """
    result = {}
    for k in group.keys():
        v = group[k]
        result[k] = dict_from_h5group(v) if isinstance(v, h5py.Group) else np.array(v)
        if isinstance(result[k], np.ndarray) and result[k].ndim == 0:
            result[k] = result[k].item()
        # Convert string arrays to strings
        if isinstance(result[k], np.ndarray) and result[k].dtype.char == "S":
            result[k] = result[k].astype(str)[0]
    return result


def h5group_to_dict(hdf5_file: h5py.File, group_name: str, data_dict: dict):
    """
    Recurses through a dictionary and creates appropriate groups or datasets
    This is parallel friendly, nothing is variable length.

    Args:
        hdf5_file: The h5py file object to write to
        group_name: The name of the group to write to
        data_dict: The dictionary to write

    Returns:
        None
    """

    for index in data_dict:
        if isinstance(data_dict[index], dict):
            h5group_to_dict(hdf5_file, f"{group_name}/{index}", data_dict[index])
        else:
            data = (
                np.array([data_dict[index]], dtype="S")
                if isinstance(data_dict[index], str)
                else data_dict[index]
            )
            hdf5_file.create_dataset(f"{group_name}/{index}", data=data)


def phonodark_to_darkmagic(
    n: dict, pp: dict, times: np.ndarray, m_chi: np.ndarray
) -> Tuple[dict, dict, dict]:
    """
    Convert the dictionaries from PhonoDark to DarkMAGIC

    Args:
        n (dict): Numerics dictionary from PhonoDark
        pp (dict): Particle physics dictionary from PhonoDark
        times (np.ndarray): Array of times
        m_chi (np.ndarray): Array of dark matter masses

    Returns:
        Tuple of Calculator, Numerics, Model dictionaries
    """
    numerics = {
        "_threshold": pp["threshold"],
        "N_grid": [n["n_a"], n["n_b"], n["n_c"]],
        "_power_abc": [n["power_a"], n["power_b"], n["power_c"]],
        "N_DWF_grid": [n["n_DW_x"], n["n_DW_y"], n["n_DW_z"]],
        "use_q_cut": n["q_cut"],
        "_use_special_mesh": n["special_mesh"],
        "bin_width": n["energy_bin_width"],
    }
    warnings.warn(
        "You are reconstructing a Calculator object from a "
        "PhonoDark formatted HDF5 file. These files do not "
        "contain the coefficient functions, so they will be set to "
        "one, which is likely incorrect. Use caution if rerunning "
        "this calculation."
    )
    coeff_func = {
        alpha: {psi: one for psi, f in c.items()} for alpha, c in pp["c_coeffs"].items()
    }

    def F_med_prop(grid):
        return grid.q_norm ** (-pp["Fmed_power"])

    model = {
        "shortname": pp.get("model_name"),
        "S_chi": pp["dm_properties"]["spin"],
        "coeff_prefactor": pp["c_coeffs"],
        "coeff_func": coeff_func,
        "F_med_prop": F_med_prop,
        "name": "Unknown Model",
    }
    calc = {
        "calc_type": "scattering",
        "m_chi": m_chi,
        "time": times,
        "v_e": None,
    }
    return calc, numerics, model
