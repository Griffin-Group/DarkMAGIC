import math
import warnings
# import itertools

import numpy as np
from numpy.typing import ArrayLike

import darkmagic.constants as const
from darkmagic.benchmark_models import BUILT_IN_MODELS
from darkmagic.io import read_h5, write_h5
from darkmagic.material import Material
from darkmagic.model import Model
from darkmagic.numerics import Numerics
from darkmagic.rate import RATE_CALC_CLASSES
from darkmagic.parallel import JOB_SENTINEL, ROOT_PROCESS, distribute_load, setup_mpi


class Calculator:
    """
    A class for calculating rates at a given list of masses and earth velocities.
    """

    def __init__(
        self,
        calc_type: str,
        m_chi: ArrayLike,
        material: Material,
        model: Model,
        numerics: Numerics,
        time: ArrayLike | None = None,
        v_e: ArrayLike | None = None,
    ):
        """
        Initializes the calculator

        Args:
            calc_type (str): The type of calculation, only "scattering" is supported at the moment
            m_chi (ArrayLike): The list of dark matter masses
            material (Material): The material
            model (Model): The model
            numerics (Numerics): The numerical parameters
            time (ArrayLike, optional): The list of times in hours. Defaults to None.
            v_e (ArrayLike, optional): The list of earth velocities. Defaults to None (i.e., calculate from the tiems)

        Raises:
            ValueError: If neither time nor v_e are provided
            ValueError: If both time and v_e are provided
        """
        if calc_type not in ["scattering"]:
            raise ValueError("Only 'scattering' is supported at the moment")

        if time is None and v_e is None:
            raise ValueError("Either time or v_e must be provided")
        if time is not None and v_e is not None:
            raise ValueError("Only one of time or v_e should be provided")

        self.v_e = self.compute_ve(time) if time is not None else v_e
        self.time = time if time is not None else np.zeros(len(self.v_e))
        self.numerics = numerics
        self.material = material
        self.m_chi = m_chi
        self.model = model
        self.comm = None  # MPI communicator

        # TODO: this is ugly
        calc_class = RATE_CALC_CLASSES.get((calc_type, type(material)))
        if calc_class is None:
            warnings.warn(
                "Material class not recognized. This is possibly because you"
                " are trying to read in a file that lacks material "
                "information (e.g., PhonoDark formatted files). "
                "If this is not the case, please report a bug."
            )

        self.calc_list = None
        if calc_class is not None:
            self.calc_list = [
                [calc_class(m, v, material, model, numerics) for m in self.m_chi]
                for v in self.v_e
            ]

        self._binned_rate, self._diff_rate, self._total_rate = None, None, None
        self.binned_rate, self.diff_rate, self.total_rate = None, None, None

    @classmethod
    def from_file(cls, filename: str, format="darkmagic"):
        """
        Load a model from a file
        """
        calc, numerics, model, data = read_h5(filename, format)
        time = calc["time"]
        m_chi = calc["m_chi"]
        numerics = Numerics.from_dict(numerics)
        model = Model.from_dict(model)
        material = None  # TODO: add material info to DarkMAGIC output
        if material is None:
            warnings.warn(
                "Material information not found in file. Running the calculation won't work, but the parsed in results can be analyzed."
            )

        calc = cls(calc["calc_type"], m_chi, material, model, numerics, time=time)
        calc.binned_rate = data["binned_rate"]
        calc.diff_rate = data["diff_rate"]
        calc.total_rate = data["total_rate"]

        return calc

    def evaluate(self, mpi: bool = False):
        """
        Computes the differential rate for all masses and earth velocities

        Args:
            mpi (bool, optional): Whether to run the calculation in parallel using MPI. Defaults to False.
        """
        nv, nm = len(self.time), len(self.m_chi)
        max_bin_num = math.ceil(self.material.max_dE / self.numerics.bin_width)
        self.diff_rate = np.zeros((nv, nm, max_bin_num))
        self.binned_rate = np.zeros((nv, nm, self.material.n_modes))
        self.total_rate = np.zeros((nv, nm))
        eval_method = self._evaluate_mpi if mpi else self._evaluate_serial
        self.comm = eval_method(
            self.diff_rate, self.binned_rate, self.total_rate, self.calc_list
        )

    def to_file(self, filename: str | None = None, format="darkmagic"):
        """
        Save the rates to a file.

        Args:
            filename (str, optional): The name of the file to save to. Defaults to None (i.e., use the default name, {material.name_model.name.h5}).
            format (str, optional): The format of the file. Defaults to "darkmagic".
        """
        if filename is None:
            filename = f"{self.material.name}_{self.model.shortname}.h5"
        # Make sure all the rates are note None
        rank = self.comm.Get_rank() if self.comm is not None else 0
        if (
            self.diff_rate is None
            or self.binned_rate is None
            or self.total_rate is None
        ) and rank == ROOT_PROCESS:
            raise ValueError(
                "Rates are not computed yet. Please run the calculation first using the evaluate method."
            )

        # TODO: re-implement parallel IO
        write_h5(
            filename,
            self.material,
            self.model,
            self.numerics,
            self.m_chi,
            self.time,
            self.v_e,
            self.total_rate,
            self.diff_rate,
            self.binned_rate,
            rank,
            None,  # should be self.comm when parallel is reimplemented
            parallel=False,
            format=format,
        )

    @staticmethod
    def compute_ve(times: float):
        """
        Returns the earth's velocity in the lab frame at time t (in hours)
        """
        theta = const.theta_earth

        v_e = np.zeros((len(times), 3))
        for it, t in enumerate(times):
            phi = 2 * np.pi * (t / 24.0)
            v_e[it] = const.VE * np.array(
                [
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta) * np.sin(theta) * (np.cos(phi) - 1),
                    (np.sin(theta) ** 2) * np.cos(phi) + np.cos(theta) ** 2,
                ]
            )
        return v_e

    def compute_daily_modulation(
        self,
        threshold_meV: float = 1.0,
    ) -> np.array:
        r"""
        Computes the rate $R$ normalized by the average daily rate $\langle R \rangle$.

        Args:
            threshold_meV (float, optional): The threshold in meV. Defaults to 1 meV.

        Returns:
            np.array: the normalized rate $R / \langle R \rangle$, indexed as (time, mass)
        """

        energy_bin_width = self.numerics.bin_width
        time = np.sort(self.time)

        # TODO: this doesn't deal with some edge cases of poorly sampled time
        t_idx = np.argwhere(time < 24)[-1][0]
        # TODO: Clarify this. Should ensure enough time points going from 0 to 23
        if (t_max := time[t_idx]) < 23:
            warnings.warn(
                f"The largest time is {t_max}. Ideally you should sample the time at 1 or 2 hour intervals."
            )

        # Vanilla PhonoDark calcs have a threshold of 1 meV by default
        # We need to account for that in the binning
        bin_cutoff = int(threshold_meV * 1e-3 / energy_bin_width) - int(
            self.numerics._threshold / energy_bin_width
        )

        # Sum the rate over the energy bins -> (time, mass) array
        total_rate = np.sum(self.diff_rate[..., bin_cutoff:], axis=2)
        # Average the rate over a full day -> (mass,) array
        day_averaged_rate = np.mean(total_rate[:t_idx, ...], axis=0)
        # Return normalized rate as a (time, mass) array
        return total_rate / day_averaged_rate

    def compute_reach(
        self,
        threshold_meV: float = 1.0,
        exposure_kg_yr: float = 1.0,
        n_cut: float = 3.0,
        model: str | None = None,
        time: float | str = 0,
    ) -> np.array:
        """
        Computes the projected reach: the 95% C.L. constraint (3 events, no background) on $\bar{sigma}_n$ or $\bar{sigma}_e$ for a given model, in units of cm2 and normalizing to the appropriate reference cross section.

        Args:
            threshold_meV (float, optional): The threshold in meV. Defaults to 1 meV.
            exposure_kg_yr (float, optional): The exposure in kg.yr. Defaults to 1 kg.yr
            n_cut (float, optional): The number of events. Defaults to 3.
            model (str, optional): The model to use. Defaults to the model attribute.
            time (float | str, optional): The time to use. Defaults to 0.

        Returns:
            np.array: the cross section.
        """

        energy_bin_width = self.numerics.bin_width
        m_chi = self.m_chi
        # get time index
        try:
            t_idx = np.argwhere(self.time == time)[0][0]
        except IndexError as e:
            raise ValueError(f"Time {time} not found in the list of times.") from e

        raw_binned_rate = 1e-100 + self.diff_rate[t_idx]  # TODO: bad names...

        # Vanilla PhonoDark calcs have a threshold of 1 meV by default
        # We need to account for that in the binning
        bin_cutoff = int(threshold_meV * 1e-3 / energy_bin_width) - int(
            self.numerics._threshold / energy_bin_width
        )

        model_name = model if model is not None else self.model.shortname
        if model_name is None:
            raise ValueError(
                "Model not provided and not found in the file. "
                "Please provide the model name."
            )
        sigma = n_cut / np.sum(raw_binned_rate[:, bin_cutoff:], axis=1)
        sigma /= (
            exposure_kg_yr
            * const.kg_yr
            * const.cm2
            * BUILT_IN_MODELS[model_name].ref_cross_sect(m_chi)
        )

        return sigma

    @staticmethod
    def _evaluate_serial(
        diff_rate: np.ndarray,
        binned_rate: np.ndarray,
        total_rate: np.ndarray,
        calc_list: list,
    ):
        """
        Evaluate the rates without MPI
        """

        nm, nv = len(calc_list[0]), len(calc_list)
        for task in np.indices((nm, nv)).T.reshape(-1, 2):
            im, iv = task[0], task[1]
            print(
                f"Rate calculation: {im + iv*nm + 1}/{(nv*nm)}.",
            )
            (
                diff_rate[iv, im],
                binned_rate[iv, im],
                total_rate[iv, im],
            ) = calc_list[iv][im].calculate_rate()

    @staticmethod
    def _evaluate_mpi(
        diff_rate: np.ndarray,
        binned_rate: np.ndarray,
        total_rate: np.ndarray,
        calc_list: list,
    ):
        """
        Evaluate the rates in parallel using MPI
        """
        comm, n_ranks, rank = setup_mpi()

        # comm, n_ranks, rank = None, 1, ROOT_PROCESS
        all_tasks = None
        if rank == ROOT_PROCESS:
            all_tasks = distribute_load(n_ranks, calc_list)

        # task_list = all_tasks[0]
        task_list = comm.scatter(all_tasks, root=ROOT_PROCESS)

        if rank == ROOT_PROCESS:
            print("Done distributing tasks.")

        diff_rates, binned_rates, total_rates = [], [], []
        # Loop over the tasks and calculate the rates
        for j, job in enumerate(task_list):
            if job[0] == JOB_SENTINEL and job[1] == JOB_SENTINEL:
                continue
            im, iv = job[0], job[1]
            print(f"Rank {rank} working on task {j+1}/{len(task_list)}")
            d, b, t = calc_list[iv][im].calculate_rate()
            diff_rates.append([job, d.real])
            binned_rates.append([job, b.real])
            total_rates.append([job, t.real])

        print(f"Rank {rank} done calculating rates.")

        # comm.Barrier()
        if rank == ROOT_PROCESS:
            print("All ranks done calculating rates. Gathering results.")
        # TODO: this is hideous....
        # Might be desirable to save these for parallel IO reimplementation
        diff_rate_list = comm.gather(diff_rates, root=ROOT_PROCESS)
        binned_rate_list = comm.gather(binned_rates, root=ROOT_PROCESS)
        total_rate_list = comm.gather(total_rates, root=ROOT_PROCESS)

        if rank == ROOT_PROCESS:
            for rate_array, rate_list in zip(
                [diff_rate, binned_rate, total_rate],
                [diff_rate_list, binned_rate_list, total_rate_list],
            ):
                for rates in rate_list:
                    for job, r in rates:
                        rate_array[job[1], job[0]] = r

        return comm
