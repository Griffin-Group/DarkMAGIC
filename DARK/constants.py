"""
    Constants in natural units (unless otherwise specified).
"""

from scipy.special import erf
import numpy as np
from numpy import pi
from scipy import constants

# Angstrom to eV, etc.
eV_to_invMeter = constants.e / (constants.h * constants.c)
eV_to_invcm = eV_to_invMeter * 1e-2
eV_to_invAng = eV_to_invMeter * 1e-10
invAng_to_eV = 1 / eV_to_invAng
invMeter_to_eV = constants.h * constants.c / constants.e
invcm_to_eV = invMeter_to_eV * 1e2
invAng_to_eV = invMeter_to_eV * 1e10
eV_to_invAng = 1 / invAng_to_eV
inveV_to_Ang = invAng_to_eV
Ang_to_inveV = 1 / inveV_to_Ang

# mass
kg_to_eV = constants.c**2 / constants.e
amu_to_eV = constants.physical_constants["atomic mass constant"][0] * kg_to_eV

# Energy and frequency
Hz_to_eV = constants.h / constants.e
s_to_inveV = 1 / Hz_to_eV
THz_to_eV = Hz_to_eV * 1e12
year_to_s = 365.25 * 24 * 60 * 60
kg_year = year_to_s * s_to_inveV * kg_to_eV

# Some constants
theta_earth = 42 * (pi / 180)  # Angle of north pole relative to earth velocity
rho_chi = 0.3e9 * invcm_to_eV**3  # DM Energy Density (1/eV^2)
alpha_FS = constants.alpha  # Fine structure constant

# Masses
m_p = constants.m_p * kg_to_eV
m_n = constants.m_n * kg_to_eV
m_e = constants.m_e * kg_to_eV

# Maxwell-Boltzmann Distribution
# These are the values in the magnon paper
V0 = 220e3 / constants.c  # Dispersion
VE = 240e3 / constants.c  # Earth velocity
VESC = 500e3 / constants.c  # Galactic escape velocity
N0 = (
    pi ** (3 / 2)
    * V0**2
    * (V0 * erf(VESC / V0) - (2 / np.sqrt(pi)) * VESC * np.exp(-(VESC**2) / V0**2))
)
C1 = pi * V0**2 / N0
C2 = np.exp(-((VESC / V0) ** 2))
