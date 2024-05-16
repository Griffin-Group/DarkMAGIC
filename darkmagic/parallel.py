"""
Module with utilities for parallel execution
"""

import numpy as np

ROOT_PROCESS = 0  # Root MPI process global
JOB_SENTINEL = -99999  # Sentinel for jobs that are not to be executed


def setup_mpi():
    """
    This function sets up the MPI environment and returns the communicator.
    """
    from mpi4py.MPI import COMM_WORLD

    n_ranks = COMM_WORLD.Get_size()  # Number of ranks
    rank = COMM_WORLD.Get_rank()  # current task rank

    if rank == ROOT_PROCESS:
        print(f"Done setting up MPI. Using {n_ranks} ranks.")
    return COMM_WORLD, n_ranks, rank


def distribute_load(n_ranks, calc_list):
    """
    This function distributes the load of jobs across the available processors.
    It attempts to balance the load as much as possible.
    """
    # TODO: implement to logging
    print("Distributing load among processors.")
    # Our task list is a list of tuples, where each tuple is a pair of (m, v) indices
    nm, nv = len(calc_list[0]), len(calc_list)
    # total_job_list = np.array(list(itertools.product(range(nm), range(nv))))
    full_task_list = np.indices((nm, nv)).T.reshape(-1, 2)
    n_jobs = len(full_task_list)

    base_jobs_per_rank = n_jobs // n_ranks  # each processor has at least this many jobs
    extra_jobs = (
        1 if n_jobs % n_ranks else 0
    )  # might need 1 more job on some processors
    # Note a fan of using a sentinel value to indicate a "do nothing" job
    task_list = JOB_SENTINEL * np.ones(
        (n_ranks, base_jobs_per_rank + extra_jobs, 2), dtype=int
    )

    for i in range(n_jobs):
        proc_index = i % n_ranks
        job_index = i // n_ranks
        task_list[proc_index, job_index] = full_task_list[i]

    # This is the number of ranks that will have on extra job
    n_unbalanced = n_jobs - n_ranks * base_jobs_per_rank
    if n_jobs > n_ranks and n_unbalanced != 0:
        print("Number of jobs exceeds the number of ranks.")
        print(f"Number of jobs per rank: {str(base_jobs_per_rank)}")
        print(f"{n_unbalanced} ranks will have on extra job each.")
    elif n_jobs < n_ranks:
        print("Number of jobs is fewer than the number of processors.")
        print("Consider reducing the number of processors for more efficiency.")
        print(f"Total number of jobs: {n_jobs}")
    else:
        print(
            "Number of jobs is perfectly divisible by the number of ranks. "
            "Optimal load balancing."
        )

    return task_list
