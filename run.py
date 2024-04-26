import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from darkmagic.io import write_output

# Get the example material and model
from darkmagic.materials.FeGe_Magnon import get_material
from darkmagic.models.magnetic_dipole import get_model
from darkmagic.numerics import Numerics
from darkmagic.parallel import JOB_SENTINEL, ROOT_PROCESS, distribute_load
from darkmagic.rate import MagnonCalculation


def main(material, model, numerics, masses, times, hdf5_filename):
    # Initialize MPI
    n_ranks = comm.Get_size()  # Number of ranks
    rank_id = comm.Get_rank()  # Rank ID

    if rank_id == ROOT_PROCESS:
        print("Done setting up MPI")

    full_job_list = None
    if rank_id == ROOT_PROCESS:
        full_job_list = distribute_load(n_ranks, masses, times)

    job_list = comm.scatter(full_job_list, root=ROOT_PROCESS)

    # TODO: Move these to numpy arrays
    diff_rate_list = []
    binned_rate_list = []
    total_rate_list = []

    if rank_id == ROOT_PROCESS:
        print("Done configuring calculation")

    # TODO: not an ideal implementation with a sentinel
    for job in job_list:
        if job[0] == JOB_SENTINEL and job[1] == JOB_SENTINEL:
            continue

        mass = masses[job[0]]
        time = times[job[1]]

        print(f"Creating calculation object for m={mass:.3f} and t={time:d}")

        calc = MagnonCalculation(mass, material, model, numerics, time=time)
        [diff_rate, binned_rate, total_rate] = calc.calculate_rate()

        diff_rate_list.append([job, np.real(diff_rate)])
        binned_rate_list.append([job, np.real(binned_rate)])
        total_rate_list.append([job, np.real(total_rate)])

    print(f"** Done computing rate on {rank_id}.")
    comm.Barrier()
    if rank_id == ROOT_PROCESS:
        print("Done on all processes, going to gather")

    write_serial = False
    if write_serial:
        all_diff_rate_list = comm.gather(diff_rate_list, root=ROOT_PROCESS)
        all_binned_rate_list = comm.gather(binned_rate_list, root=ROOT_PROCESS)
        all_total_rate_list = comm.gather(total_rate_list, root=ROOT_PROCESS)
    else:
        all_diff_rate_list = [diff_rate_list]
        all_binned_rate_list = [binned_rate_list]
        all_total_rate_list = [total_rate_list]

    write_output(
        hdf5_filename,
        material,
        model,
        numerics,
        masses,
        times,
        all_total_rate_list,
        all_diff_rate_list,
        all_binned_rate_list,
        rank_id,
        comm,
        parallel=True,
    )

    if rank_id == ROOT_PROCESS:
        print("Done writing rate.")


if __name__ == "__main__":
    masses = np.logspace(4, 7, 90)
    times = [0]
    material = get_material()
    model = get_model()
    numerics = Numerics(N_grid=[40, 20, 20], use_special_mesh=False)
    hdf5_filename = f"out/DarkMAGIC_{material.name}_{model.name}.hdf5"

    main(material, model, numerics, masses, times, hdf5_filename)
