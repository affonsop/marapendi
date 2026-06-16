"""
Cell model: explicit steady-state PEMFC performance model.

:class:`ExplicitSteadyStateModel` orchestrates the solve sequence:
heat-transfer resistance → MEA temperature → water transport →
gas concentrations → cell voltage. The voltage sub-calculations are
delegated to :class:`~marapendi.voltage.VoltageModel`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cantera as ct

from .electrochemistry import h2_lhv
from .voltage import VoltageModel


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
        fc.calculate_heat_transfer_resistance()
        if mea_temperature_estimation:
            fc.set_mea_temperature(fc.temperature)
            fc.calculate_water_transport()
            fc.calculate_gas_concentrations_at_cl()
            v0 = self.voltage_model.compute_cell_voltage(fc)
            mea_temperature = fc.temperature + (
                fc.current_density * (-h2_lhv(fc.temperature) / (2 * ct.faraday) - v0)
                * fc.thermal_resistance
            )
        else:
            mea_temperature = fc.temperature + (fc.current_density * 0.7) * fc.thermal_resistance

        fc.set_mea_temperature(mea_temperature)
        fc.calculate_water_transport()
        fc.calculate_gas_concentrations_at_cl()
        self.voltage_model.compute_cell_voltage(fc)
        return fc.cell_voltage
