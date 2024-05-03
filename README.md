# DarkMAGIC 🔮
[![Develop Docs](https://img.shields.io/badge/status-pre--alpha-red)](https://oashour.github.io/DarkMAGIC/develop/)
![Tests](https://github.com/oashour/DarkMAGIC/actions/workflows/run_tests.yaml/badge.svg)
[![License](https://img.shields.io/badge/License-MIT-blue)](#license "Go to license section")
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue)](https://Griffin-Group.github.io/DarkMAGIC/latest/)
[![Develop Docs](https://img.shields.io/badge/docs-develop-purple)](https://Griffin-Group.github.io/DarkMAGIC/develop/)
<!--![GitHub Release](https://img.shields.io/github/v/release/oashour/DarkMAGIC?include_prereleases)-->

The **D**ark **M**atter ***A****b* *initio* phonon/ma**G**non **I**nteraction **C**alculator (**DarkMAGIC**) is a python package for calculating dark matter interaction rates with phonons and magnons in real materials, based on properties calculated from first principles (mainly, density functional theory). It is based on [the effective field theory developed by Trickle, Zhang and Zurek](https://arxiv.org/abs/2009.13534), and takes inspiration from the code they released with that paper, [PhonoDark](https://github.com/tanner-trickle/PhonoDark). 

DarkMAGIC is currently in pre-alpha testing, so not all features are implemented and not everything has been tested. Development is in progress, and once at a satisfying stage, a manuscript will be prepared.

## Features
* Calculate scattering rates from single phonons and magnons.
* Supports phonon calculations using the frozen cell method or density functional perturbation theory with arbitrary DFT codes, based on the interface with [phonopy](https://phonopy.github.io/phonopy/).
* Supports toy models and *ab initio*-based spin hamiltonians via [rad-tools](https://rad-tools.org/en/stable/) and [TB2J](https://tb2j.readthedocs.io/en/latest/).
* Parallelized with MPI, and extremely performant.
* Easy to use Python API, with multiple pre-defined benchmark models.
* Portable HDF5 output format that allows the reconstruction of calculation as python objects. DarkMAGIC can also read and write HDF5 files in the format used by PhonoDark, but with limited functionality since it does not include all the parameters necessary to rebuild a calculation.

### Planned Features (short term)
* In principle, all operators in the [EFT paper](https://arxiv.org/abs/2009.13534) are implemented for phonons, but only spin-independent operators are currently functional due to a recent refactoring. Only the magnetic dipole and anapole models for magnons are currently implemented. This will be dealt with soon.
* JIT compilation for increased performance.
* Command line interface.
* Further post-processing and plotting tools.
* More documentation and examples. This website contains extensive, automatically-generated documentation for the python API.

### Planned Features (long term)
* Multi-phonon processes
* Absorption
* Support for fully first-principles time-dependent DFT magnon calculations instead of just spin Hamiltonians.

# Installation
DarkMAGIC will soon be made available on pip. For now, please install it as

```shell
pip install git+https://github.com/oashour/DarkMAGIC
```
