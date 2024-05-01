import h5py
import numpy as np

import darkmagic.constants as const
from darkmagic.benchmark_models import BUILT_IN_MODELS


def get_reach(
    filename: str,
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
        model (str, optional): The model to use. Defaults to None (find it in file)
        time (float | str, optional): The time to use. Defaults to 0.

    Returns:
        np.array: the cross section.
    """

    with h5py.File(filename, "r") as f:
        energy_bin_width = f["numerics"]["energy_bin_width"][...]
        threshold = f["particle_physics"]["threshold"][...]
        m_chi = np.array(f["particle_physics"]["dm_properties"]["mass_list"])
        raw_binned_rate = 1e-100 + np.array(  # 1e-100 to avoid division by zero
            [
                np.array(f["data"]["diff_rate"][str(time)][str(m)])
                for m in range(len(m_chi))
            ]
        )
        # TODO: add check to see if model is there

    # Vanilla PhonoDark, have a threshold of 1 meV by default
    # We need to account for that in the binning
    bin_num_cut = int(threshold_meV * 1e-3 / energy_bin_width) - int(
        threshold / energy_bin_width
    )

    cs_constraint = n_cut / np.sum(raw_binned_rate[:, bin_num_cut:], axis=1)
    cs_constraint /= (
        exposure_kg_yr
        * const.kg_yr
        * const.cm2
        * BUILT_IN_MODELS[model].ref_cross_sect(m_chi)
    )

    return cs_constraint
