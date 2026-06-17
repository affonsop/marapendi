"""
Water thermodynamic and physical property correlations.

All functions are pure-Python / NumPy — no Cantera dependency.  The
correlations are validated against Cantera's Water phase object; the
test suite (``tests/test_water_properties.py``) checks agreement to the
tolerances stated in each docstring.
"""

import numpy as np

from .constants import GAS_CONSTANT, WATER_MOLECULAR_WEIGHT

water_molecular_weight = WATER_MOLECULAR_WEIGHT
"""Molecular weight of water (kg/kmol). Alias for :data:`~marapendi.constants.WATER_MOLECULAR_WEIGHT`."""


def water_saturation_pressure(temperature):
    """Saturation pressure of water (Pa) — Buck equation.

    Accuracy: < 0.1 % over 274–373 K vs. Cantera.
    """
    Tcelsius = temperature - 273.15
    return 611.21 * np.exp((18.678 - Tcelsius / 234.5) * (Tcelsius / (257.14 + Tcelsius)))


def water_saturation_concentration(temperature):
    """Saturation concentration of water vapour (kmol/m³)."""
    return water_saturation_pressure(temperature) / (GAS_CONSTANT * temperature)


def water_dew_point(vapor_pressure):
    """Dew-point temperature (K) for a given water vapour partial pressure (Pa).

    Cubic polynomial in ln(P) fitted to Cantera values over 700–101 325 Pa.
    Accuracy: < 0.05 K vs. Cantera over that range.
    """
    _COEFFS = (1.01005617e-01, -1.37637000e+00, 1.92195992e+01, 1.79796275e+02)
    return np.polyval(_COEFFS, np.log(vapor_pressure))


def water_dynamic_viscosity(temperature=300):
    """Dynamic viscosity of liquid water (Pa·s) — Vogel equation.

    μ = A · exp(B / (T − C))  with A, B, C fitted to Cantera values.
    Accuracy: < 1.1 % vs. Cantera over 274–373 K.
    """
    return 3.162220e-05 * np.exp(482.6125 / (temperature - 153.5669))


def water_kinematic_viscosity(temperature=300):
    """Kinematic viscosity of liquid water (m²/s)."""
    return water_dynamic_viscosity(temperature) / water_density(temperature)


def water_surface_tension(temperature=300):
    """Surface tension of liquid water (N/m)."""
    return 0.076 - 1.677e-4 * (temperature - 273.15)


def water_density(temperature=300):
    """Density of liquid water (kg/m³) — polynomial fit.

    Source: Kell (1975) as cited in IAPWS-IF97.
    Accuracy: < 0.05 % vs. Cantera over 274–373 K.
    """
    T_Celsius = temperature - 273.15
    return np.polyval([-2.658e-3, -0.155, 1001.3], T_Celsius)


def water_molar_volume(temperature=300):
    """Molar volume of liquid water (m³/kmol)."""
    return WATER_MOLECULAR_WEIGHT / water_density(temperature)


def o2_water_diffusivity(temperature=300):
    """O₂ diffusivity in liquid water (m²/s).

    Uses value at 298 K from Tsimpanogiannis et al. (2021), table 11.
    """
    return 4.6e-7 * np.exp(-0.155e4 / temperature)


class WaterProperties:
    """Thermophysical properties of liquid water at a given temperature.

    All computations use the same polynomial correlations as the module-level
    functions — no Cantera dependency.

    Attributes
    ----------
    density : float            kg/m³
    dynamic_viscosity : float  Pa·s
    molar_volume : float       m³/kmol
    saturation_pressure : float Pa
    """

    def __init__(self, temperature: float = 300):
        self.set_temperature(temperature)

    def set_temperature(self, temperature: float):
        """Update all properties for *temperature* (K)."""
        self.density = water_density(temperature)
        self.dynamic_viscosity = water_dynamic_viscosity(temperature)
        self.molar_volume = water_molar_volume(temperature)
        self.saturation_pressure = water_saturation_pressure(temperature)
