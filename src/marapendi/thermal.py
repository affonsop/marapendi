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
from .water import water_molar_volume


@dataclass
class ThermalModel:
    """Computes the heat transfer resistance and MEA temperature of a :class:`Cell`."""

    def side_heat_transfer_resistance(self, side: CellSide) -> float:
        """Heat transfer resistance on one side between the MEA and the channel (m²·K/W)."""
        return (
            sum(layer.thermal_resistance() for layer in side.porous_layers if layer is not side.cl)
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
        from .electrochemistry import h2_lhv
        from .constants import FARADAY_CONSTANT

        thermal_resistance = self.heat_transfer_resistance(cell)
        if mea_temperature_estimation:
            v0 = state.cell_voltage
            return state.temperature + (
                state.current_density
                * (-h2_lhv(state.temperature) / (2 * FARADAY_CONSTANT) - v0)
                * thermal_resistance
            )
        return state.temperature + (state.current_density * 0.7) * thermal_resistance

    def set_mea_temperature(self, mea_temperature: float, cell, state) -> None:
        """Write the MEA temperature onto *state* and the catalyst-layer components of *cell*."""
        state.mea_temperature = mea_temperature
        state.mea_temperature_increase = mea_temperature - state.temperature
        state.membrane.temperature = mea_temperature
        state.mea_water_molar_volume = water_molar_volume(mea_temperature)
        state.ca.cl.temperature = mea_temperature
        state.an.cl.temperature = mea_temperature
        # Keep component objects in sync so methods called on them see the right temperature.
        cell.ca.cl.set_gas_temperature(mea_temperature)
        cell.an.cl.set_gas_temperature(mea_temperature)
        cell.membrane.temperature = mea_temperature
        cell.mea_water_molar_volume = water_molar_volume(mea_temperature)
