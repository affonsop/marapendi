"""
Liquid water properties, cached per state.

:class:`WaterModel` provides stateless correlations for liquid water
properties (surface tension, kinematic viscosity, density) as pure
functions of ``state.temperature``, cached on the ``state`` object to
avoid recomputing them.
"""
from __future__ import annotations

from ..legacy.water import water_density, water_kinematic_viscosity, water_surface_tension


class WaterModel:
    """Stateless correlations for liquid water properties.

    All methods take a ``state`` object exposing ``temperature`` (K) --
    typically a :class:`~marapendi.future.state.LayerState` or
    :class:`~marapendi.future.state.MembraneState` -- and cache their
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
