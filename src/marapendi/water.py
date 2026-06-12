"""
Liquid water and water vapor properties.

Free functions provide thermodynamic and physical properties of water
(based on Cantera's ``Water`` object and standard correlations).
:class:`WaterModel` wraps the liquid-water property functions with
stateless correlations as pure functions of ``state.temperature``, cached
on the ``state`` object to avoid recomputing them.

References
----------
https://cantera.org/documentation/docs-3.0/sphinx/html/cython/importing.html#cantera.Water
"""
from __future__ import annotations

import cantera as ct
import numpy as np

h2o_phase = ct.Water()

water_molecular_weight = h2o_phase.molecular_weights[0]


def water_saturation_pressure(temperature):
    """Saturation pressure of water (Pa) at ``temperature`` (K)."""
    Tcelsius = temperature - 273.15
    return 611.21 * np.exp((18.678 - Tcelsius / 234.5) * (Tcelsius / (257.14 + Tcelsius)))


def water_dynamic_viscosity(temperature=300):
    """Dynamic viscosity of water (Pa.s) at ``temperature`` (K)."""
    h2o = ct.SolutionArray(h2o_phase, np.shape(temperature))
    h2o.TQ = temperature, 0  # Set temperature and vapor quality
    return h2o.viscosity


def water_kinematic_viscosity(temperature=300):
    """Kinematic viscosity of water (m^2/s) at ``temperature`` (K)."""
    return water_dynamic_viscosity(temperature) / water_density(temperature)


def water_surface_tension(temperature=300):
    """Surface tension of water (N/m) at ``temperature`` (K)."""
    return 0.076 - 1.677e-4 * (temperature - 273.15)


def water_density(temperature=300):
    """Density of water (kg/m^3) at ``temperature`` (K).

    Source: https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470516430.app3
    """
    T_Celsius = temperature - 273.15
    return np.polyval([-2.658e-3, -0.155, 1001.3], T_Celsius)


def water_molar_volume(temperature=300):
    """Molar volume of water (m^3/kmol) at ``temperature`` (K)."""
    return h2o_phase.molecular_weights[0] / water_density(temperature)


def o2_water_diffusivity(temperature=300):
    """O2 diffusivity in liquid water (m^2/s) at ``temperature`` (K).

    Uses the value at 298 K from Tsimpanogiannis et al. (2021), table 11.
    """
    return 4.6e-7 * np.exp(-0.155e4 / temperature)


class WaterModel:
    """Stateless correlations for liquid water properties.

    All methods take a ``state`` object exposing ``temperature`` (K) --
    typically a :class:`~marapendi.state.LayerState` or
    :class:`~marapendi.state.MembraneState` -- and cache their
    result on the state.
    """

    @staticmethod
    def surface_tension(state) -> float:
        """Surface tension of water at ``state.temperature`` (N/m)."""
        if state.surface_tension is None:
            state.surface_tension = water_surface_tension(state.temperature)
        return state.surface_tension

    @staticmethod
    def kinematic_viscosity(state) -> float:
        """Kinematic viscosity of water at ``state.temperature`` (m^2/s)."""
        if state.kinematic_viscosity is None:
            state.kinematic_viscosity = water_kinematic_viscosity(state.temperature)
        return state.kinematic_viscosity

    @staticmethod
    def density(state) -> float:
        """Density of water at ``state.temperature`` (kg/m^3)."""
        if state.density is None:
            state.density = water_density(state.temperature)
        return state.density
