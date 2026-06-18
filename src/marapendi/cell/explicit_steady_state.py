"""
Cell model: explicit steady-state PEMFC performance model.

:class:`ExplicitSteadyStateModel` orchestrates the solve sequence:
heat-transfer resistance → MEA temperature → water transport →
gas concentrations → cell voltage.

Each sub-model receives both the component tree (``cell``) for static
physics parameters and a :class:`~marapendi.state.CellState` (``state``)
for runtime values.  The model reads from and writes to ``state``; component
objects in ``cell`` are kept in sync for backward compatibility.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from ..thermo.constants import FARADAY_CONSTANT
from ..thermo.gas import GasModel, species_indexes
from .thermal import ThermalModel
from .gas_transport import GasTransportModel
from .voltage import VoltageModel
from .water_balance import MembraneWaterBalanceModel
from .state import CellState, LayerState, CatalystLayerState


@dataclass
class ExplicitSteadyStateModel:
    """
    Explicit steady-state model for PEMFC polarization curves.

    Hypotheses
    ----------
    - MEA temperature increases linearly with current density (or is estimated
      from the first-pass cell voltage when ``mea_temperature_estimation=True``).
    - Membrane water transport uses dry-condition gas-transport resistances.
    - Liquid water saturation follows the Darcy power-law from the net flux.
    """

    voltage_model: VoltageModel = field(default_factory=VoltageModel)
    thermal_model: ThermalModel = field(default_factory=ThermalModel)
    water_balance_model: MembraneWaterBalanceModel = field(default_factory=MembraneWaterBalanceModel)
    gas_transport_model: GasTransportModel = field(default_factory=GasTransportModel)

    def set_initial_state(self, cell, stack_temperature, current_density,
                          cathode_conditions, anode_conditions) -> CellState:
        """Create and initialise a fresh :class:`CellState` for one operating point.

        Returns a new state each call so that array shapes always match
        *current_density* without manually zeroing pre-existing arrays.

        Parameters
        ----------
        cell : FuelCell
            Component tree — provides geometry and static physics objects.
        stack_temperature : float or ndarray
            Operating temperature of the fuel-cell stack (K).
        current_density : float or ndarray
            Current density (A/m²).
        cathode_conditions : OperatingConditions
            Inlet conditions for the cathode side.
        anode_conditions : OperatingConditions
            Inlet conditions for the anode side.

        Returns
        -------
        CellState
            Fully initialised state ready for :meth:`solve`.
        """
        state = CellState()
        for side, side_state in zip(cell.sides, state.sides):
            if side.has_mpl:
                side_state.mpl = LayerState()

        state.current_density = current_density
        self._set_consumption_production(state, current_density)

        state.temperature = stack_temperature
        state.membrane.temperature = stack_temperature

        for side, side_state in zip(cell.sides, state.sides):
            for layer_state in side_state.layers:
                layer_state.temperature = stack_temperature

            side.cl.set_water_film_thickness(0)
            side_state.is_liquid_equilibrated = False

        for side, side_state, conditions in zip(
            cell.sides, state.sides, (cathode_conditions, anode_conditions)
        ):
            side_state.cl.ionomer_water_content = 10
            side.cl.set_ionomer_wet_properties(side_state.cl.ionomer_water_content, stack_temperature)
            side.electrolyte = conditions.inlet_liquid
            side.electrolyte.set_temperature(stack_temperature)

            n = np.atleast_1d(current_density).size
            for layer_state in side_state.layers:
                layer_state.pressure = 0.5 * (conditions.inlet_pressure + conditions.outlet_pressure)
                layer_state.gas.X = np.zeros((n, 4))
                GasModel.set_composition(
                    layer_state,
                    conditions.dry_o2_mole_fraction,
                    conditions.dry_h2_mole_fraction,
                    conditions.inlet_relative_humidity,
                    conditions.inlet_pressure,
                    conditions.inlet_temperature,
                )

        self._set_flow_rates(cell, state, cathode_conditions, anode_conditions)

        for side, side_state in zip(cell.sides, state.sides):
            for component, component_state in zip(side.layers, side_state.layers):
                component.electrolyte = side.electrolyte
                component.update_state_at_temperature(component_state, stack_temperature)
                if component.contact_angle < 90:
                    component_state.non_wetting_saturation = (
                        1 - side.ch.inlet_liquid_saturation * np.ones_like(current_density)
                    )
                else:
                    component_state.non_wetting_saturation = (
                        side.ch.inlet_liquid_saturation * np.ones_like(current_density)
                    )

        return state

    def _set_consumption_production(self, state, current_density) -> None:
        """Write O₂/H₂ consumption and H₂O production rates into *state*."""
        state.ca.reactant_consumption = current_density / (4 * FARADAY_CONSTANT)
        state.an.reactant_consumption = current_density / (2 * FARADAY_CONSTANT)
        state.ca.h2o_production = current_density / (2 * FARADAY_CONSTANT)
        state.an.h2o_production = 0

    def _set_flow_rates(self, cell, state, cathode_conditions, anode_conditions) -> None:
        """Compute and store inlet gas/liquid flow rates on the channel state."""
        for cell_side, side_state, conditions in zip(
            cell.sides, state.sides, (cathode_conditions, anode_conditions)
        ):
            side_state.ch.inlet_liquid_flow_rate = conditions.inlet_liquid_flow_rate
            side_state.ch.inlet_gas_flow_rate = (
                conditions.stoichiometry
                * side_state.reactant_consumption
                * cell.area
                / (
                    side_state.ch.gas.X[..., species_indexes[cell_side.reactant]]
                    * GasModel.concentration(side_state.ch)
                )
                + conditions.inlet_gas_flow_rate
            )
            side_state.ch.inlet_liquid_saturation = (
                side_state.ch.inlet_liquid_flow_rate
                / np.maximum(
                    side_state.ch.inlet_liquid_flow_rate + side_state.ch.inlet_gas_flow_rate,
                    1e-12,
                )
            )

    def solve(self, cell, state, mea_temperature_estimation: bool = False) -> float:
        """
        Run the explicit steady-state solve on *cell* / *state* and return the cell voltage.

        Parameters
        ----------
        cell : FuelCell
            Fully configured fuel-cell object.  Provides static physics parameters.
        state : CellState
            Runtime state returned by :meth:`set_initial_state`.  All computed
            quantities are written here.
        mea_temperature_estimation : bool
            When ``True``, estimate the MEA temperature from a first-pass
            voltage calculation instead of the 0.7 V efficiency approximation.

        Returns
        -------
        float or ndarray
            Cell voltage (V). The value is stored on ``state.cell_voltage``.
        """
        state.thermal_resistance = self.thermal_model.heat_transfer_resistance(cell)

        if mea_temperature_estimation:
            self.thermal_model.set_mea_temperature(state.temperature, cell, state)
            self.water_balance_model.calculate_water_transport(cell, state)
            self.gas_transport_model.calculate_gas_concentrations(cell, state)
            self.voltage_model.compute_cell_voltage(cell, state)

        mea_temperature = self.thermal_model.mea_temperature(cell, state, mea_temperature_estimation)
        self.thermal_model.set_mea_temperature(mea_temperature, cell, state)
        self.water_balance_model.calculate_water_transport(cell, state)
        self.gas_transport_model.calculate_gas_concentrations(cell, state)
        self.voltage_model.compute_cell_voltage(cell, state)
        return state.cell_voltage
