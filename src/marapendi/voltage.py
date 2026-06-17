"""
Voltage model: reversible voltage, overpotentials and resulting cell voltage.

:class:`VoltageModel` computes all voltage quantities for a fuel cell,
reading and writing state directly on the fuel-cell object and its components.
This separates voltage physics from the fuel-cell component tree and from the
model orchestration in :class:`marapendi.model.ExplicitSteadyStateModel`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import FARADAY_CONSTANT
from .electrochemistry import calculate_reversible_cell_voltage, STD_PRESSURE
from .water import water_molar_volume


@dataclass
class VoltageModel:
    """Stateless collection of voltage calculations for a :class:`~marapendi.fuelcell.FuelCell`."""

    def reversible_cell_voltage(self, fc) -> float:
        activity_o2 = fc.ca.cl.species_partial_pressure('o2') / STD_PRESSURE
        activity_h2 = fc.an.cl.species_partial_pressure('h2') / STD_PRESSURE
        return calculate_reversible_cell_voltage(
            fc.ca.cl.temperature, activity_o2 ** 0.5 * activity_h2,
        )

    def reversible_voltage_vs_RHE(self, fc) -> float:
        return calculate_reversible_cell_voltage(
            fc.ca.cl.temperature,
            fc.ca.cl.species_partial_pressure('o2') / STD_PRESSURE,
        )

    def activation_overpotential(self, fc, theta_PtO: float = 0) -> float:
        fc.h2_permeation_flux = fc.membrane.hydrogen_permeation_flux(
            fc.an.cl.species_partial_pressure('h2'),
            fc.membrane.temperature,
            fc.an.cl.pressure - fc.ca.cl.pressure,
            fc.membrane.water_vol_fraction(fc.membrane.water_content, fc.mea_water_molar_volume),
        )
        fc.crossover_current = fc.h2_permeation_flux * (2 * FARADAY_CONSTANT)
        omega_PtO_voltage_drop = (
            fc.ca.cl.omega_PtO * theta_PtO
            / (fc.ca.cl.reaction.number_of_electrons * fc.ca.cl.reaction.charge_transfer_coeff * FARADAY_CONSTANT)
        )
        fc.orr_overpotential = fc.ca.cl.reaction.tafel_overpotential(
            (fc.current_density + fc.crossover_current)
            / (fc.ca.cl.ecsa * fc.ca.cl.platinum_loading * (1 - theta_PtO)),
            fc.ca.cl.temperature,
            fc.ca.cl.species_partial_pressure('o2'),
        )
        fc.hor_overpotential = 0
        return fc.orr_overpotential + omega_PtO_voltage_drop + fc.hor_overpotential

    def high_frequency_resistance(self, fc) -> float:
        return (
            fc.membrane.proton_resistance(fc.membrane.temperature, water_saturation=fc.ca.cl.liquid_saturation)
            + fc.electrical_resistance
        )

    def ohmic_overpotential(self, fc) -> float:
        side = fc.ca
        side.cl.proton_resistance = 1. / (
            side.cl.non_wetting_saturation / side.cl.effective_charge_resistance(
                fc.current_density, side.liquid_eq_water_content, side.cl.temperature,
            ) + (1 - side.cl.non_wetting_saturation) / side.cl.effective_charge_resistance(
                fc.current_density, side.vapor_eq_water_content, side.cl.temperature,
            )
        )
        return fc.current_density * (side.cl.proton_resistance + self.high_frequency_resistance(fc))

    def calculate_theta_PtO(self, fc) -> float:
        E_rev_vs_RHE = self.reversible_voltage_vs_RHE(fc)
        theta_PtO = 0
        if fc.ca.cl.omega_PtO > 0:
            eps_max = 10
            eta_act = 0
            while eps_max > 0.001:
                eta_act_old = eta_act
                eta_act = self.activation_overpotential(fc, theta_PtO)
                theta_PtO = 0.5 * theta_PtO + 0.5 / (1 + np.exp(22.4 * (0.818 - E_rev_vs_RHE + eta_act)))
                eps_max = np.mean(np.abs(eta_act - eta_act_old))
        fc.ca.cl.theta_catalyst = theta_PtO
        return theta_PtO

    def compute_cell_voltage(self, fc) -> float:
        """Compute the cell voltage, writing E_rev, eta_act, eta_ohm and cell_voltage onto *fc*."""
        theta_PtO = self.calculate_theta_PtO(fc)
        E_rev = self.reversible_cell_voltage(fc)
        eta_ohm = self.ohmic_overpotential(fc)
        eta_act = self.activation_overpotential(fc, theta_PtO)
        fc.cell_voltage = np.maximum(0, E_rev - eta_act - eta_ohm)
        return fc.cell_voltage
