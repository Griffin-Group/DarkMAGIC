import itertools

import numpy as np


ROOT_PROCESS = 0  # Root MPI process global
JOB_SENTINEL = -99999


def distribute_load(n_proc, masses, times):
    """
    This function distributes the load of jobs across the available processors.
    It attempts to balance the load as much as possible.
    """
    # TODO: change to logging
    print("Distributing load among processors.")
    # Our jobs list is a list of tuples, where each tuple is a pair of (mass, time) indices
    total_job_list = np.array(
        list(itertools.product(range(len(masses)), range(len(times))))
    )
    n_jobs = len(total_job_list)

    base_jobs_per_proc = n_jobs // n_proc  # each processor has at least this many jobs
    extra_jobs = 1 if n_jobs % n_proc else 0  # might need 1 more job on some processors
    # Note a fan of using a sentinel value to indicate a "do nothing" job
    job_list = JOB_SENTINEL * np.ones(
        (n_proc, base_jobs_per_proc + extra_jobs, 2), dtype=int
    )

    for i in range(n_jobs):
        proc_index = i % n_proc
        job_index = i // n_proc
        job_list[proc_index, job_index] = total_job_list[i]

    if n_jobs > n_proc:
        print("Number of jobs exceeds the number of processors.")
        print("Baseline number of jobs per processor: " + str(base_jobs_per_proc))
        print(
            "Remaining processors with one extra job: "
            + str(n_jobs - n_proc * base_jobs_per_proc)
        )
    elif n_jobs < n_proc:
        print("Number of jobs is fewer than the number of processors.")
        print("Consider reducing the number of processors for more efficiency.")
        print("Total number of jobs: " + str(n_jobs))
    else:
        print("Number of jobs matches the number of processors. Maximally parallized.")

    return job_list
