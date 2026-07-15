"""
Explicit steady-state PEMFC performance model.

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
from ..thermo.gas import species_indexes
from ..thermal import ThermalModel
from ..gas_transport_resistance import GasTransportModel
from ..voltage import VoltageModel
from ..water_balance.water_balance import WaterBalanceModel
from ...simulation.state import CellState, LayerState, CatalystLayerState, GasFlowState


@dataclass
class ExplicitSteadyStateModel:
    """
    Explicit steady-state model for PEMFC polarization curves.

    Usage
    -----
    ::

        model = ExplicitSteadyStateModel()
        conditions = CellConditions(
            current_density=np.linspace(1e3, 2e4, 20),
            cell_temperature=353.15,
            ca=SideConditions(outlet_pressure=1.5e5, dry_o2_mole_fraction=0.21, ...),
            an=SideConditions(outlet_pressure=1.5e5, dry_h2_mole_fraction=1.0, ...),
        )
        state = model.set_initial_conditions(cell, conditions)
        state = model.solve(cell, conditions, state)
        # state.cell_voltage, state.mea_temperature, … are now populated

    Notes
    -----
    - MEA temperature is estimated analytically: either from the 0.7 V efficiency
      approximation (``mea_temperature_estimation=False``) or from a first-pass
      voltage calculation (``mea_temperature_estimation=True``).
    - Membrane water transport uses dry-condition gas-transport resistances.
    - Liquid water saturation follows the Darcy power-law from the net flux.

    Parameters
    ----------
    mea_temperature_estimation : bool
        When ``True``, perform a first-pass voltage calculation at the stack
        temperature before estimating T_MEA, giving a better approximation
        than the constant 0.7 V efficiency assumption.
    """

    voltage_model: VoltageModel = field(default_factory=VoltageModel)
    thermal_model: ThermalModel = field(default_factory=ThermalModel)
    water_balance_model: WaterBalanceModel = field(default_factory=WaterBalanceModel)
    gas_transport_model: GasTransportModel = field(default_factory=GasTransportModel)

    def set_initial_conditions(self, cell, cell_conditions) -> CellState:
        """Create and initialise a :class:`CellState` from *cell_conditions*.

        Returns a fresh state object with gas compositions, flow rates and
        initial temperatures set from the conditions.  Pass the result to
        :meth:`solve`.

        Parameters
        ----------
        cell : FuelCell
            Component tree — provides geometry and static physics objects.
        cell_conditions : CellConditions
            Operating conditions: current density, stack temperature, and one
            :class:`~marapendi.simulation.conditions.SideConditions` per side.

        Returns
        -------
        CellState
        """
        return self._init_state(
            cell,
            cell_conditions.cell_temperature,
            cell_conditions.current_density,
            cell_conditions.ca,
            cell_conditions.an,
        )

    def solve(self, cell, cell_conditions, initial_state: CellState) -> CellState:
        """Run the explicit steady-state solve and return the populated state.

        Parameters
        ----------
        cell : FuelCell
            Fully configured fuel-cell object.
        cell_conditions : CellConditions
            Operating conditions (same object passed to :meth:`set_initial_conditions`).
        initial_state : CellState
            State returned by :meth:`set_initial_conditions`.  Modified in place.

        Returns
        -------
        CellState
            The same object as *initial_state*, now containing all solved
            quantities (``cell_voltage``, ``mea_temperature``, transport
            resistances, gas concentrations, …).
        """
        state = initial_state
        state.thermal_resistance = self.thermal_model.heat_transfer_resistance(cell)

        mea_temperature = self.thermal_model.mea_temperature(
            cell, state, None
        )
        self.thermal_model.set_mea_temperature(mea_temperature, cell, state)
        self.water_balance_model.calculate_water_transport(
            cell, state, gas_transport_model=self.gas_transport_model
        )
        self.gas_transport_model.calculate_gas_concentrations(cell, state)
        self.voltage_model.compute_cell_voltage(cell, state)

        for cell_side, side_state, conditions in zip(
            cell.sides, state.sides, (cell_conditions.ca, cell_conditions.an)
        ):
            self.set_gas_flow_states(cell, cell_side, side_state, conditions, cell_conditions.cell_temperature)

        return state

    def set_gas_flow_states(self, cell, cell_side, side_state, side_conditions,
                             stack_temperature: float) -> None:
        """Populate ``side_state.inlet_gas_flow_state``/``outlet_gas_flow_state``.

        Call once ``side_state.reactant_consumption``/``h2o_production`` (set by
        :meth:`_set_consumption_production`) and ``side_state.vapor_flux``/``liquid_flux``
        (set by :meth:`~marapendi.models.water_balance.water_balance.WaterBalanceModel.update_cell_side_water_fluxes`,
        itself called from ``calculate_water_transport``) are available — i.e.
        after :meth:`~marapendi.models.water_balance.water_balance.WaterBalanceModel.calculate_water_transport`
        has run. Also called by :class:`~marapendi.models.base.transient.TransientModel`
        through its internal :class:`ExplicitSteadyStateModel` instance.

        ``GasFlowState`` models a single operating point (its fields are plain
        floats, not arrays), so this is a no-op — leaving both fields ``None`` —
        when *side_state* comes from a vectorised solve (e.g. a polarization
        curve with an array-valued ``current_density``). Call once per scalar
        operating point to get flow states.

        Parameters
        ----------
        cell : FuelCell
        cell_side : FuelCellSide
            ``cell.ca`` or ``cell.an`` — used for ``cell_side.ch.reactant``.
        side_state : CellSideState
            ``state.ca`` or ``state.an``. Modified in place.
        side_conditions : SideConditions
        stack_temperature : float
            ``cell_conditions.cell_temperature``.
        """
        if np.size(side_state.reactant_consumption) != 1:
            return

        reactant = cell_side.ch.reactant
        n_electrons = 4 if cell_side is cell.ca else 2
        minimal_reactant_consumption = side_conditions.minimum_current_density_for_stoich / (n_electrons * FARADAY_CONSTANT)
        reactant_consumption = float(np.asarray(side_state.reactant_consumption).reshape(()))
        stack_temperature = float(np.asarray(stack_temperature).reshape(()))

        side_state.inlet_gas_flow_state = GasFlowState.from_side_conditions(
            side_conditions, stack_temperature, reactant,
            reactant_consumption, minimal_reactant_consumption, cell.area,
        )
        side_state.outlet_gas_flow_state = side_state.inlet_gas_flow_state.consume(
            reactant, reactant_consumption,
            float(np.asarray(side_state.vapor_flux).reshape(())),
            float(np.asarray(side_state.liquid_flux).reshape(())),
            cell.area,
        )

    # ------------------------------------------------------------------
    # Internal helpers (also called by FuelCell for legacy support)
    # ------------------------------------------------------------------

    def _init_state(self, cell, stack_temperature, current_density,
                    cathode_conditions, anode_conditions) -> CellState:
        """Low-level state initialiser used by :meth:`set_initial_conditions`."""
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
                layer_state.gas.set_composition(conditions.dry_o2_mole_fraction,
                    conditions.dry_h2_mole_fraction,
                    conditions.inlet_relative_humidity,
                    conditions.inlet_pressure,
                    conditions.inlet_temperature,)

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
        state.an.reactant_consumption = state.ca.reactant_consumption * 2
        state.ca.h2o_production = state.an.reactant_consumption
        state.an.h2o_production = 0
        
    def _set_flow_rates(self, cell, state, cathode_conditions, anode_conditions) -> None:
        """Compute and store inlet gas/liquid flow rates on the channel state."""
        for cell_side, side_state, conditions in zip(
            cell.sides, state.sides, (cathode_conditions, anode_conditions)
        ):
            minimal_reactant_consumption = conditions.minimum_current_density_for_stoich / ((4 if cell_side is cell.ca else 2) * FARADAY_CONSTANT)
            side_state.ch.inlet_liquid_flow_rate = conditions.inlet_liquid_flow_rate
            side_state.ch.inlet_gas_flow_rate = (
                conditions.stoichiometry
                * np.maximum(side_state.reactant_consumption, minimal_reactant_consumption)
                * cell.area
                / (
                    side_state.ch.gas.X[..., species_indexes[cell_side.reactant]]
                    * side_state.ch.gas.concentration()
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

