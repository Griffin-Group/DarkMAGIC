import h5py
import numpy as np


def write_group_from_dict(hdf5_file, group_name, data_dict):
    """
    Recurses through a dictionary and creates appropriate groups or datasets
    This is parallel friendly, nothing is variable length.
    """

    for index in data_dict:
        if type(data_dict[index]) is dict:
            # This should probably be write_dict_parallel?
            write_group_from_dict(hdf5_file, f"{group_name}/{index}", data_dict[index])
        else:
            data = (
                np.array([data_dict[index]], dtype="S")
                if isinstance(data_dict[index], str)
                else data_dict[index]
            )
            hdf5_file.create_dataset(f"{group_name}/{index}", data=data)


def hdf5_write_output_parallel(
    out_file,
    numerics_parameters,
    physics_parameters,
    dm_properties_dict,
    c_dict,
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
    """

    # Get appropriate context manager for serial/parallel
    if comm is None:
        cm = h5py.File(out_file, "w")
    else:
        cm = h5py.File(out_file, "w", driver="mpio", comm=comm)

    with cm as out_f:
        # Create groups/datasets and write out input parameters
        write_group_from_dict(out_f, "numerics", numerics_parameters)
        write_group_from_dict(out_f, "particle_physics", physics_parameters)
        out_f.create_dataset("version", data=np.array(["1.1.0"], dtype="S"))
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