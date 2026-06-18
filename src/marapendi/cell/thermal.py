"""
Thermal model: heat transfer resistance and MEA temperature.

:class:`ThermalModel` computes the MEA heat transfer resistance and estimates
the MEA operating temperature from the current density and stack temperature.
It writes temperatures back onto the state object so that subsequent transport
and voltage calculations see the correct values.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cell import Cell, CellSide
from ..thermo.water import water_molar_volume, water_saturation_pressure


@dataclass
class ThermalModel:
    """Computes the heat transfer resistance and MEA temperature of a :class:`Cell`."""

    def side_heat_transfer_resistance(self, side: CellSide) -> float:
        """Heat transfer resistance on one side between the MEA and the channel (m²·K/W)."""
        return (
            sum(layer.thermal_resistance for layer in side.porous_layers if layer is not side.cl)
            + side.thermal_contact_resistance
        )

    def heat_transfer_resistance(self, cell: Cell) -> float:
        """Overall MEA heat transfer resistance (m²·K/W).

        Cathode and anode sides are parallel thermal resistances.
        """
        return 1. / sum(1. / self.side_heat_transfer_resistance(side) for side in cell.sides)

    def mea_temperature(self, cell, state, mea_temperature_estimation: bool = False) -> float:
        """Estimate the MEA temperature.

        When *mea_temperature_estimation* is ``False`` (default), assumes a
        constant 0.7 V efficiency approximation.  When ``True``, uses a
        first-pass cell voltage already stored on *state* as ``state.cell_voltage``.
        """
        from ..thermo.electrochemistry import h2_lhv
        from ..thermo.constants import FARADAY_CONSTANT

        thermal_resistance = self.heat_transfer_resistance(cell)
        if mea_temperature_estimation:
            heat_release = state.current_density * (
                -h2_lhv(state.temperature) 
                / (2 * FARADAY_CONSTANT) 
                - state.cell_voltage
            )
        else:  
            heat_release = (state.current_density * 0.7)
        return state.temperature + heat_release * thermal_resistance

    def calculate_heat_transport(self, cell, dynamic: bool = False) -> None:
        """Compute heat transport parameters and write results onto *cell*.

        Sets ``cell.thermal_resistance`` and ``cell.heat_release_rate``.
        When *dynamic* is ``False``, also sets ``cell.mea_temperature_increase``
        from the steady-state balance; when ``True``, skips that update so the
        transient integrator can evolve the temperature independently.
        """
        from ..thermo.electrochemistry import h2_lhv
        from ..thermo.constants import FARADAY_CONSTANT

        thermal_resistance = self.heat_transfer_resistance(cell)
        cell.heat_release_rate = (
            -h2_lhv(cell.temperature) / (2 * FARADAY_CONSTANT) - cell.cell_voltage
        ) * cell.current_density
        if not dynamic:
            cell.mea_temperature_increase = cell.heat_release_rate * thermal_resistance

    def temperature_rate_of_change(self, cell) -> float:
        """Compute dT/dt for the MEA temperature in a transient simulation."""
        self.calculate_heat_transport(cell, dynamic=True)
        return (
            cell.heat_release_rate
            - cell.mea_temperature_increase / cell.thermal_resistance
        ) / cell.mea_surface_heat_capacity

    def set_mea_temperature(self, mea_temperature: float, cell, state) -> None:
        """Write the MEA temperature onto *state* and the catalyst-layer components of *cell*."""
        state.mea_temperature = mea_temperature
        state.mea_temperature_increase = mea_temperature - state.temperature
        state.mea_water_molar_volume = water_molar_volume(mea_temperature)
        # Compute saturation pressure once and store on each state layer.
        mea_saturation_pressure = water_saturation_pressure(mea_temperature)
        for layer in (state.membrane, state.ca.cl, state.an.cl):
            layer.temperature = mea_temperature
            layer.saturation_pressure = mea_saturation_pressure
        # Sync temperature and the precomputed saturation pressure to component CLs
        # so legacy water-balance code reading from the component tree is correct.
        for cl in (cell.ca.cl, cell.an.cl):
            cl.temperature = mea_temperature
            cl.saturation_pressure = mea_saturation_pressure
        cell.membrane.temperature = mea_temperature
        cell.mea_water_molar_volume = water_molar_volume(mea_temperature)
