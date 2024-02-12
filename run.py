from mpi4py import MPI
import numpy as np
import itertools


import DARK.parallel as parallel
from DARK.rate import MagnonCalculation
from DARK.core import Numerics
from DARK.io import write_hdf5

# Get the example material and model
from DARK.materials.VBTS_Magnon import get_material
from DARK.models.anapole import get_model

def get_numerics():
    return Numerics(N_abc=[20, 10, 10], use_special_mesh=False)

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    n_proc = comm.Get_size()  # Number of ranks
    proc_id = comm.Get_rank()  # Processor ID

    # ID of root process
    root_process = 0

    if proc_id == root_process:
        print("Done setting up MPI")

    job_list = None
    job_list_recv = None
    masses = np.logspace(4, 7, 60)
    times = [0]
    material = get_material()
    model = get_model()
    numerics = get_numerics()

    if proc_id == root_process:

        print("Configuring calculation ...\n")

        num_jobs = len(masses) * len(times)
        print("  Total number of jobs : " + str(num_jobs))
        print()

        total_job_list = [
            (m, t)
            for (m, t) in itertools.product(range(len(masses)), range(len(times)))
        ]
        job_list = parallel.generate_job_list(n_proc, np.array(total_job_list))

        print("Going to scatter")

    job_list_recv = comm.scatter(job_list, root=root_process)

    diff_rate_list = []
    binned_rate_list = []
    total_rate_list = []

    if proc_id == root_process:
        print("Done configuring calculation")

    for job in range(len(job_list_recv)):

        if job_list_recv[job, 0] != -1 and job_list_recv[job, 1] != -1:

            job_id = job_list_recv[job]

            mass = masses[int(job_id[0])]
            time = times[int(job_id[1])]

            if proc_id == root_process:
                print(f"Creating calculation object for m={mass:.3f} and {time:d}")

            calc = MagnonCalculation(mass, material, model, numerics, time=time)
            [diff_rate, binned_rate, total_rate] = calc.calculate_rate()

            diff_rate_list.append([job_list_recv[job], np.real(diff_rate)])
            binned_rate_list.append([job_list_recv[job], np.real(binned_rate)])
            total_rate_list.append([job_list_recv[job], np.real(total_rate)])

    print(f"********** Done computing rate on {proc_id}.")
    if proc_id == root_process:
        print(
            "Done computing rate on root. Returning all data to root node to write."
        )
    comm.Barrier()
    if proc_id == root_process:
        print("Done on all processes, going to gather")

    write_serial = False
    out_filename = f"out/DARK_{material.name}_{model.name}.hdf5"
    print(out_filename)
    if write_serial:
        # return data back to root
        all_diff_rate_list = comm.gather(diff_rate_list, root=root_process)
        all_binned_rate_list = comm.gather(binned_rate_list, root=root_process)
        all_total_rate_list = comm.gather(total_rate_list, root=root_process)

        # write to output file
        if proc_id == root_process:
            print("Done gathering!!!")
            print("----------")

            write_hdf5(
                out_filename,
                material,
                numerics,
                model,
                masses,
                times,
                all_total_rate_list,
                all_diff_rate_list,
                all_binned_rate_list,
            )
    else:
        # See doc string for explanation of why the lists
        # are being put into one element lists
        write_hdf5(
            out_filename,
            material,
            numerics,
            model,
            masses,
            times,
            [total_rate_list],
            [diff_rate_list],
            [binned_rate_list],
            comm=comm,
        )

    if proc_id == root_process:
        print("Done writing rate.")


if __name__ == "__main__":
    main()
