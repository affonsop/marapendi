"""
Thermal model: heat transfer resistance and MEA temperature.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cell import Cell, CellSide
from .state import CellState
from ..legacy.water import water_saturation_pressure

@dataclass
class ThermalModel:
    """Computes the heat transfer resistance and MEA temperature of a :class:`Cell`."""

    def side_heat_transfer_resistance(self, side: CellSide) -> float:
        """Heat transfer resistance between the MEA and the channel, on one side (m^2 K / W)."""
        return (
            sum(layer.thermal_resistance() for layer in side.porous_layers if layer is not side.cl)
            + side.thermal_contact_resistance
        )

    def heat_transfer_resistance(self, cell: Cell) -> float:
        """Overall heat transfer resistance of the MEA (m^2 K / W).

        The cathode and anode sides act as parallel thermal resistances.
        """
        return 1. / sum(1. / self.side_heat_transfer_resistance(side) for side in cell.sides)

    def mea_temperature(self, cell: Cell, state: CellState) -> float:
        """Estimate of the MEA temperature, assuming a fixed fraction of the cell
        voltage loss is converted to heat (eq. ``explicit_steady_state_model``,
        ``mea_temperature_estimation=False``)."""
        thermal_resistance = self.heat_transfer_resistance(cell)
        return state.temperature + (state.current_density * 0.7) * thermal_resistance

    def set_mea_temperature(self, mea_temperature: float, state: CellState) -> None:
        """Set the MEA temperature on the membrane and catalyst layers of ``state``.

        The membrane and the catalyst layers are assumed to be at the same temperature.
        """
        for layer in (state.membrane, state.ca.cl, state.an.cl):
            layer.temperature = mea_temperature

        mea_saturation_pressure = water_saturation_pressure(mea_temperature)
        state.membrane.saturation_pressure = mea_saturation_pressure
        for side in state.sides:
            side.cl.saturation_pressure = mea_saturation_pressure