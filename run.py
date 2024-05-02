import numpy as np

# from mpi4py.MPI import COMM_WORLD as comm
from darkmagic import MaterialParameters, PhononMaterial

# Get the example material and model
from darkmagic.benchmark_models import (
    heavy_scalar_mediator,
    light_scalar_mediator,
)

# magnetic_dipole,
from darkmagic.calculator import Calculator

# from darkmagic.materials.VBTS_Magnon import get_material
from darkmagic.numerics import Numerics

# Masses and times
masses = np.logspace(4, 10, 96)
# masses = [1e5, 1e7]
times = [0]

# Phonons in Helium
params = MaterialParameters(N={"e": [2, 2], "n": [2, 2], "p": [2, 2]})
material = PhononMaterial("hcp_He", params, "tests/data/hcp_He_1GPa.phonopy.yaml")
model = light_scalar_mediator
model = heavy_scalar_mediator

# Magnons in VBTS
# material = get_material()
# model = magnetic_dipole

# Numerics
numerics = Numerics(
    N_grid=[80, 40, 40],
    N_DWF_grid=[30, 30, 30],
    use_special_mesh=False,
    use_q_cut=True,
)
# hdf5_filename = f"out/DarkMAGIC_{material.name}_{model.shortname}_whatever.hdf5"
full_calc = Calculator("scattering", masses, material, model, numerics, times)
full_calc.evaluate()
full_calc.to_file()

# main(material, model, numerics, masses, times, hdf5_filename)
print("actually done")
