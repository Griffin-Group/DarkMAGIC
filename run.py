import numpy as np
from mpi4py.MPI import COMM_WORLD as comm

from darkmagic.io import write_h5

# Get the example material and model
from darkmagic.benchmark_models import magnetic_dipole
from darkmagic.materials.VBTS_Magnon import get_material

from darkmagic.numerics import Numerics
from darkmagic.parallel import JOB_SENTINEL, ROOT_PROCESS, distribute_load
from darkmagic.calculator import Calculator


def main(material, model, numerics, masses, times, hdf5_filename):
    # Initialize MPI
    n_ranks = comm.Get_size()  # Number of ranks
    rank_id = comm.Get_rank()  # Rank ID

    if rank_id == ROOT_PROCESS:
        print("Done setting up MPI")

    full_job_list = None
    if rank_id == ROOT_PROCESS:
        full_job_list = distribute_load(n_ranks, masses, times)
    full_calc = Calculator("scattering", masses, material, model, numerics, times)

    job_list = comm.scatter(full_job_list, root=ROOT_PROCESS)

    # TODO: Move these to numpy arrays
    diff_rate_list = []
    binned_rate_list = []
    total_rate_list = []

    if rank_id == ROOT_PROCESS:
        print("Done configuring calculation")

    # Very ugly...
    # if isinstance(material, PhononMaterial):
    #    calc_class = PhononCalculation
    # elif isinstance(material, MagnonMaterial):
    #    calc_class = MagnonCalculation
    # else:
    #    raise ValueError("Material not recognized.")
    # TODO: not an ideal implementation with a sentinel
    for job in job_list:
        if job[0] == JOB_SENTINEL and job[1] == JOB_SENTINEL:
            continue

        mass = masses[job[0]]
        time = times[job[1]]

        print(f"Creating calculation object for m={mass:.3f} and t={time:d}")

        # calc = calc_class(mass, material, model, numerics, time=time)
        calc = full_calc.calc_list[job[1]][job[0]]
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

    write_h5(
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
    # Masses and times
    masses = np.logspace(4, 7, 8)
    masses = [1e5, 1e6, 1e7]
    times = [0, 1]

    # Phonons in Helium
    # params = MaterialParameters(N={"e": [2, 2], "n": [2, 2], "p": [2, 2]})
    # material = PhononMaterial("hcp_He", params, "tests/data/hcp_He_1GPa.phonopy.yaml")
    # model = heavy_scalar_mediator

    # Magnons in VBTS
    material = get_material()
    model = magnetic_dipole

    # Numerics
    numerics = Numerics(
        N_grid=[
            5,
            1,
            1,
        ],
        N_DWF_grid=[4, 4, 4],
        use_special_mesh=False,
        use_q_cut=True,
    )
    hdf5_filename = f"out/DarkMAGIC_{material.name}_{model.shortname}_whatever.hdf5"
    # Serial implementation
    full_calc = Calculator("scattering", masses, material, model, numerics, times)
    full_calc.evaluate()
    full_calc.to_file()

    # main(material, model, numerics, masses, times, hdf5_filename)
