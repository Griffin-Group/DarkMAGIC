import h5py
import numpy as np

from darkmagic.parallel import ROOT_PROCESS


def write_output(
    out_filename,
    material,
    model,
    numerics,
    masses,
    times,
    all_total_rate_list,
    all_diff_rate_list,
    all_binned_rate_list,
    proc_id,
    comm,
    parallel=False,
):
    # Write to file
    # TODO: is there a more succinct way to do this?
    print(f"Writing to file {out_filename}{' in parallel' if parallel else ''}...")
    if not parallel and proc_id == ROOT_PROCESS:
        print("Done gathering!!!")
        print("----------")

        write_hdf5(
            out_filename,
            material,
            model,
            numerics,
            masses,
            times,
            all_total_rate_list,
            all_diff_rate_list,
            all_binned_rate_list,
            comm=None,
        )
    else:
        write_hdf5(
            out_filename,
            material,
            model,
            numerics,
            masses,
            times,
            all_total_rate_list,
            all_diff_rate_list,
            all_binned_rate_list,
            comm=comm,
        )


def write_group_from_dict(hdf5_file, group_name, data_dict):
    """
    Recurses through a dictionary and creates appropriate groups or datasets
    This is parallel friendly, nothing is variable length.
    """

    for index in data_dict:
        if isinstance(data_dict[index], dict):
            write_group_from_dict(hdf5_file, f"{group_name}/{index}", data_dict[index])
        else:
            data = (
                np.array([data_dict[index]], dtype="S")
                if isinstance(data_dict[index], str)
                else data_dict[index]
            )
            hdf5_file.create_dataset(f"{group_name}/{index}", data=data)


# TODO: should be a method of the future "Full Calculation" class
def write_hdf5(
    out_file,
    material,
    model,
    numerics,
    masses,
    times,
    all_total_rate_list,
    all_diff_rate_list,
    all_binned_rate_list,
    comm=None,
):
    """

    Write data to hdf5 file

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
    - out_file: Path to the output HDF5 file.
    - material: Material object.
    - model: Model object.
    - numerics: Numerics object.
    - masses: List of masses for the jobs.
    - times: List of times for the jobs.
    - all_total_rate_list: List of total rate data.
    - all_diff_rate_list: List of differential rate data.
    - all_binned_rate_list: List of binned rate data.
    - comm: MPI communicator for parallel writing (default is None).

    Returns:
    - None
    """

    def get_dicts(model, numerics, masses, times):
        physics_parameters = {
            # energy threshold
            "threshold": 0,
            # time of days (hr)
            "times": times,
            # - d log FDM / d log q. q dependence of mediator propagator
            "Fmed_power": model.Fmed_power,
            # power of q in the potential, used to find optimal integration mesh
            "power_V": model.power_V,
            # flag to compute for a specific model
            # SI computes using the algorithm presented in 1910.08092
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
            "power_a": numerics.power_abc[0],
            "power_b": numerics.power_abc[1],
            "power_c": numerics.power_abc[2],
            "n_DW_x": numerics.N_DWF_grid[0],
            "n_DW_y": numerics.N_DWF_grid[1],
            "n_DW_z": numerics.N_DWF_grid[2],
            "energy_bin_width": numerics.bin_width,
            "q_cut": numerics.use_q_cut,
            "special_mesh": numerics.use_special_mesh,
        }
        return physics_parameters, dm_properties_dict, coeff, numerics_parameters

    physics_parameters, dm_properties_dict, c_dict, numerics_parameters = get_dicts(
        model, numerics, masses, times
    )

    # Get appropriate context manager for serial/parallel
    if comm is None:
        cm = h5py.File(out_file, "w")
    else:
        cm = h5py.File(out_file, "w", driver="mpio", comm=comm)

    with cm as out_f:
        # Create groups/datasets and write out input parameters
        write_group_from_dict(out_f, "numerics", numerics_parameters)
        write_group_from_dict(out_f, "particle_physics", physics_parameters)
        out_f.create_dataset("version", data=np.array(["0.0.1"], dtype="S"))
        write_group_from_dict(
            out_f, "particle_physics/dm_properties", dm_properties_dict
        )
        write_group_from_dict(out_f, "particle_physics/c_coeffs", c_dict)

        num_bins = len(all_diff_rate_list[0][0][1])
        num_modes = len(all_binned_rate_list[0][0][1])
        # In parallel all datasets need to be created by all ranks
        for i in range(len(physics_parameters["times"])):
            for j in range(len(dm_properties_dict["mass_list"])):
                out_f.create_dataset(f"data/rate/{i}/{j}", shape=(1,), dtype="f8")
                out_f.create_dataset(
                    f"data/diff_rate/{i}/{j}", shape=(num_bins,), dtype="f8"
                )
                out_f.create_dataset(
                    f"data/binned_rate/{i}/{j}", shape=(num_modes,), dtype="f8"
                )
        # Write out rates to the file
        # To accomodate for serial, can flatten all_total_rate_list
        for tr_list, br_list, dr_list in zip(
            all_total_rate_list, all_binned_rate_list, all_diff_rate_list
        ):
            for t, b, d in zip(tr_list, br_list, dr_list):
                # The first element of t, b, and d is a tuple of the mass and time index
                # The second element is either a scalar (total) or an array (binned and diff)
                # For binned and diff, the dimensions of the array are num_bins and num_modes
                # respectively

                # mass_index, time_index
                j, i = map(str, map(int, t[0]))
                # print(f'Writing data/rate/{i}/{j} = {t[1]} to {out_f["data"]["rate"][i][j]}')
                out_f["data"]["rate"][i][j][...] = t[1]
                out_f["data"]["binned_rate"][i][j][...] = b[1]
                out_f["data"]["diff_rate"][i][j][...] = d[1]
