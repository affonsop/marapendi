"""
Cell model orchestration.

:class:`CellModel` operates on a :class:`~marapendi.future.cell.Cell` and a
:class:`~marapendi.future.state.CellState` to compute the cell's behaviour.

This module currently only defines the base class and its method
signatures. The implementation will be added component by component.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..legacy.water import water_saturation_pressure
from .cell import Cell
from .conditions import CellOperatingConditions
from .constants import FARADAY_CONSTANT
from .gas import GasModel
from .state import CellState
from .thermal import ThermalModel
from .transport import GasTransportModel
from .voltage import VoltageModel
from .water_balance import MembraneWaterBalanceModel


@dataclass
class CellModel:
    """Base class orchestrating calculations on a :class:`Cell`."""

    water_balance_model: MembraneWaterBalanceModel = field(default_factory=MembraneWaterBalanceModel)

    def steady_state_solution(self, conditions):
        """Compute the steady-state solution of the cell for given operating ``conditions``."""
        pass

    def transient_solution(self, conditions):
        """Compute the transient solution of the cell for given operating ``conditions``."""
        pass


@dataclass
class ExplicitSteadyStateModel(CellModel):
    """Simplified steady-state model where all calculations are explicit.

    Reproduces :meth:`marapendi.legacy.fuelcell.FuelCell.explicit_steady_state_model`
    with ``mea_temperature_estimation=False``.
    """

    cell: Cell = field(default_factory=Cell)
    thermal_model: ThermalModel = field(default_factory=ThermalModel)
    gas_transport_model: GasTransportModel = field(default_factory=GasTransportModel)
    voltage_model: VoltageModel = field(default_factory=VoltageModel)

    def steady_state_solution(self, conditions) -> CellState:
        state = self.initial_state(conditions)
        mea_temperature = self.thermal_model.mea_temperature(self.cell, state)
        self.thermal_model.set_mea_temperature(mea_temperature, state)

        self.water_balance_model.calculate_water_transport(self.cell, state, self.gas_transport_model)
        self.gas_transport_model.calculate_gas_concentrations(self.cell, state)
        self.voltage_model.compute_cell_voltage(self.cell, state)
        return state

    def initial_state(self, conditions: CellOperatingConditions) -> CellState:
        """Initialize a :class:`CellState` from the operating ``conditions``."""
        state = CellState()
        state.current_density = conditions.current_density
        state.temperature = conditions.cell_temperature
        state.membrane.temperature = conditions.cell_temperature
        cell_saturation_pressure = water_saturation_pressure(conditions.cell_temperature)
        side_reactant = {'ca': 'o2', 'an': 'h2'}
        o2_consumption = state.current_density / (4 * FARADAY_CONSTANT)
        side_reactant_consumption = {'ca': o2_consumption, 'an': 2 * o2_consumption}
        side_state_h2o_production = {'ca': 2 * o2_consumption, 'an': 0.}
        for side_name, side_state in zip(('ca', 'an'), state.sides):
            side_conditions = getattr(conditions, side_name)
            side_state.cl.theta_catalyst = 0.
            side_state.h2o_production = side_state_h2o_production[side_name]
            for layer in side_state.layers:
                layer.temperature = conditions.cell_temperature
                layer.pressure = side_conditions.pressure
                layer.saturation_pressure = cell_saturation_pressure
                layer.liquid_saturation = 0.
                layer.non_wetting_saturation = np.zeros_like(state.current_density, dtype=float)
                GasModel.set_composition(
                    layer,
                    side_conditions.dry_o2_mole_fraction,
                    side_conditions.dry_h2_mole_fraction,
                    side_conditions.relative_humidity,
                )
            reactant = side_reactant[side_name]
            side_state.ch.inlet_gas_flow_rate = (
                side_conditions.stoichiometry * side_reactant_consumption[side_name] * self.cell.area
                / GasModel.species_mole_fraction(side_state.ch, reactant)
                / GasModel.concentration(side_state.ch)
            )
        return state
