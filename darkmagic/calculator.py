import math
import warnings

import numpy as np
from mpi4py.MPI import COMM_WORLD
from numpy.typing import ArrayLike

import darkmagic.constants as const
from darkmagic.benchmark_models import BUILT_IN_MODELS
from darkmagic.io import read_h5, write_h5
from darkmagic.material import Material
from darkmagic.model import Model
from darkmagic.numerics import Numerics
from darkmagic.parallel import JOB_SENTINEL, ROOT_PROCESS, distribute_load
from darkmagic.rate import RATE_CALC_CLASSES


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

    def evaluate(self):
        """
        Computes the differential rate for all masses and earth velocities
        """

        n_ranks = COMM_WORLD.Get_size()  # Number of ranks
        rank = COMM_WORLD.Get_rank()  # Processor rank

        if rank == ROOT_PROCESS:
            print("Done setting up MPI")

        all_tasks = None
        if rank == ROOT_PROCESS:
            all_tasks = distribute_load(n_ranks, self.m_chi, self.v_e)

        task_list = COMM_WORLD.scatter(all_tasks, root=ROOT_PROCESS)

        if rank == ROOT_PROCESS:
            print("Done configuring calculation")

        # Set up empty arrays to store the results
        max_bin_num = math.ceil(self.material.max_dE / self.numerics.bin_width)
        self.diff_rate = np.zeros((len(self.time), len(self.m_chi), max_bin_num))
        self.binned_rate = np.zeros(
            (len(self.time), len(self.m_chi), self.material.n_modes)
        )
        self.total_rate = np.zeros((len(self.time), len(self.m_chi)))

        # Loop over the tasks and calculate the rates
        for job in task_list:
            if job[0] == JOB_SENTINEL and job[1] == JOB_SENTINEL:
                continue
            im, iv = job[0], job[1]
            (
                self.diff_rate[iv, im],
                self.binned_rate[iv, im],
                self.total_rate[iv, im],
            ) = self.calc_list[iv][im].calculate_rate()
        print(f"Rank {rank} done calculating rates.")
        COMM_WORLD.Barrier()

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
        if (
            self.diff_rate is None
            or self.binned_rate is None
            or self.total_rate is None
        ):
            raise ValueError(
                "Rates are not computed yet. Please run the calculation first using the evaluate method."
            )
        write_h5(
            filename,
            self.material,
            self.model,
            self.numerics,
            self.m_chi,
            self.time,  # TODO: generate this even if zeros
            self.v_e,
            self.total_rate,
            self.diff_rate,
            self.binned_rate,
            COMM_WORLD.Get_rank(),
            COMM_WORLD,
            parallel=False,
            format=format,
        )  # Need to implement the parallel writing with the class...

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
            filename (str): The path to the HDF5 file containing the data.
            threshold_meV (float, optional): The threshold in meV. Defaults to 1.0.
            exposure_kg_yr (float, optional): The exposure in kg.yr. Defaults to 1.0.
            n_cut (float, optional): The number of events. Defaults to 3.0.
            model (str, optional): The model to use. Defaults to None (i.e., finds the model name in the file)
            time (float | str, optional): The time to use. Defaults to 0.

        Returns:
            np.array: the cross section.
        """

        energy_bin_width = self.numerics.bin_width
        threshold = self.numerics._threshold  # legacy for PD support
        print("got a threshold of ", threshold)
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
            threshold / energy_bin_width
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
