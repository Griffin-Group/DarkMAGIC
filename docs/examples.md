# Examples

More examples coming soon

## Light Scalar Mediator, Single Phonons, Solid He


This can be run from a Jupyter notebook or as a python script
```python
import numpy as np

from darkmagic import Calculator, MaterialParameters, PhononMaterial, Numerics
from darkmagic.benchmark_models import light_scalar_mediator

# Masses in eV and times of day in hours (to calculate the earth's velocity)
masses = np.logspace(4, 10, 96)
times = [0]

# Phonons in Helium
params = MaterialParameters(N={"e": [2, 2], "n": [2, 2], "p": [2, 2]})
material = PhononMaterial("hcp_He", params, "tests/data/hcp_He_1GPa.phonopy.yaml")
model = light_scalar_mediator

# Numerics
numerics = Numerics(
    N_grid=[80, 40, 40],  # Spherical grid for momentum transfer
    N_DWF_grid=[30, 30, 30],  # Monkhorst-Pack grid for Debye-Waller factor
)

# Create calculator object
full_calc = Calculator("scattering", masses, material, model, numerics, times)
full_calc.evaluate()  # Run calculation
full_calc.to_file()  # Write HDF5 file (default name is "material.name_model.name.h5")
```

To run in parallel, simply run with `srun -n <nprocs> python script.py` or whatever alternative your system uses, and use `full_calc.evaluate(mpi=True)` instead. Note that you need to install the optional `mpi` dependency for MPI calculations to work (i.e., `pip install darkmagic[mpi]`).