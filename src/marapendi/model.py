"""
Cell model: explicit steady-state PEMFC performance model.

:class:`ExplicitSteadyStateModel` orchestrates the solve sequence:
heat-transfer resistance → MEA temperature → water transport →
gas concentrations → cell voltage. The voltage sub-calculations are
delegated to :class:`~marapendi.voltage.VoltageModel`; the thermal
sub-calculations are delegated to :class:`~marapendi.thermal.ThermalModel`.
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

    def solve(self, fc, mea_temperature_estimation: bool = False) -> float:
        """
        Run the explicit steady-state solve on *fc* and return the cell voltage.

        Parameters
        ----------
        fc : FuelCell
            Fully configured fuel-cell object (conditions already set via
            ``fc.set_conditions``).
        mea_temperature_estimation : bool
            When ``True``, estimate the MEA temperature from a first-pass
            voltage calculation instead of the 0.7 V efficiency approximation.

        Returns
        -------
        float or ndarray
            Cell voltage (V). The value is also stored on ``fc.cell_voltage``.
        """
        fc.thermal_resistance = self.thermal_model.heat_transfer_resistance(fc)
        if mea_temperature_estimation:
            self.thermal_model.set_mea_temperature(fc.temperature, fc)
            self.water_balance_model.calculate_water_transport(fc)
            self.gas_transport_model.calculate_gas_concentrations(fc)
            self.voltage_model.compute_cell_voltage(fc)

        mea_temperature = self.thermal_model.mea_temperature(fc, mea_temperature_estimation)
        self.thermal_model.set_mea_temperature(mea_temperature, fc)
        self.water_balance_model.calculate_water_transport(fc)
        self.gas_transport_model.calculate_gas_concentrations(fc)
        self.voltage_model.compute_cell_voltage(fc)
        return fc.cell_voltage
