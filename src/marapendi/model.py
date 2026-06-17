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

from dataclasses import dataclass, field

from .thermal import ThermalModel
from .transport import GasTransportModel
from .voltage import VoltageModel
from .water_balance_models import MembraneWaterBalanceModel


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

    def solve(self, cell, state, mea_temperature_estimation: bool = False) -> float:
        """
        Run the explicit steady-state solve on *cell* / *state* and return the cell voltage.

        Parameters
        ----------
        cell : FuelCell
            Fully configured fuel-cell object (conditions already set via
            ``fc.set_conditions``).  Provides static physics parameters.
        state : CellState
            Runtime state populated by ``fc.populate_state()``.  All computed
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
