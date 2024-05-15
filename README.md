# DarkMAGIC 🔮
[![Pre-Alpha](https://img.shields.io/badge/Status-Pre--Alpha-red)](https://Griffin-Group.github.io/DarkMAGIC/develop/)
[![GitHub Release](https://img.shields.io/github/v/release/Griffin-Group/DarkMAGIC?include_prereleases)](https://github.com/Griffin-Group/DarkMAGIC/releases)
[![Tests](https://github.com/oashour/DarkMAGIC/actions/workflows/run_tests.yaml/badge.svg)](https://github.com/Griffin-Group/DarkMAGIC/actions)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license "Go to license section")
[![Stable Docs](https://img.shields.io/badge/Docs-Stable-blue)](https://Griffin-Group.github.io/DarkMAGIC/latest/)
[![Develop Docs](https://img.shields.io/badge/Docs-Develop-purple)](https://Griffin-Group.github.io/DarkMAGIC/develop/)
[![DOI](https://zenodo.org/badge/755802517.svg)](https://zenodo.org/doi/10.5281/zenodo.11124265)

The **D**ark **M**atter ***A****b* *initio* phonon/ma**G**non **I**nteraction **C**alculator (DarkMAGIC) is a python package for calculating dark matter interaction rates with phonons and magnons in real materials, based on properties calculated from first principles (mainly, density functional theory). It is based on [the effective field theory developed by Trickle, Zhang and Zurek](https://arxiv.org/abs/2009.13534), and takes inspiration from the code they released with that paper and its precursors, [PhonoDark](https://github.com/tanner-trickle/PhonoDark). 

DarkMAGIC is currently in pre-alpha testing, so not all features are implemented, the public API is subject to change, and not everything has been tested thoroughly. Development is in progress, and once at a satisfying stage, a manuscript will be prepared.

## Features
* Calculate scattering rates, reach and daily modulation for single phonons and magnons.
* Supports phonon calculations using the frozen cell method or density functional perturbation theory with arbitrary DFT codes, based on the interface with [phonopy](https://phonopy.github.io/phonopy/).
* Supports magnon calculations using *ab initio*-based spin hamiltonians via [rad-tools](https://rad-tools.org/en/stable/) and [TB2J](https://tb2j.readthedocs.io/en/latest/).
* Parallelized with MPI, and highly performant.
* Easy to use Python API, with multiple pre-defined benchmark models.
* Portable HDF5 output format that allows the reconstruction of calculation as python objects. DarkMAGIC can also read and write HDF5 files in the format used by PhonoDark, but with limited functionality since it does not include all the parameters necessary to rebuild a calculation.

### Planned Features (short term)
* In principle, all operators in the [EFT paper](https://arxiv.org/abs/2009.13534) are implemented for phonons, but only spin-independent operators are currently functional due to a recent refactoring. Only the magnetic dipole and anapole models for magnons are currently implemented. This will be dealt with soon.
* JIT compilation and shared-memory parallelism for increased performance.
* Command line interface.
* Plotting tools
* More documentation and examples. This website contains extensive, automatically-generated documentation for the python API, but it is not yet complete.

### Planned Features (long term)
* Multi-phonon processes
* Absorption
* Support for fully first-principles time-dependent DFT magnon calculations instead of just spin Hamiltonians.

# Installation
DarkMAGIC will soon be made available on pip. For now, please install it as

```shell
pip install git+https://github.com/Griffin-Group/DarkMAGIC
```


# Citation
If you use DarkMAGIC in your research, please cite it as

```bibtex
@misc{darkmagic,
  author       = {Omar A. Ashour and Sin\'{e}ad M. Griffin},
  title        = { {DarkMAGIC} },
  month        = May,
  year         = 2024,
  doi          = {10.5281/zenodo.11124265},
  url          = {https://dx.doi.org/10.5281/zenodo.11124265}
}
```