"""
Voltage model: reversible voltage, overpotentials and resulting cell voltage.

:class:`VoltageModel` computes all voltage quantities for a fuel cell,
reading gas-phase state via :class:`~marapendi.gas.GasModel` from the
:class:`~marapendi.state.CellState` and reading static physics parameters
from the component tree (``cell``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import FARADAY_CONSTANT
from ..electrochemistry import calculate_reversible_cell_voltage, STD_PRESSURE
from ..gas import GasModel
from ..water import water_molar_volume


@dataclass
class VoltageModel:
    """Stateless collection of voltage calculations for a fuel cell."""

    def reversible_cell_voltage(self, cell, state) -> float:
        activity_o2 = GasModel.species_partial_pressure(state.ca.cl, 'o2') / STD_PRESSURE
        activity_h2 = GasModel.species_partial_pressure(state.an.cl, 'h2') / STD_PRESSURE
        return calculate_reversible_cell_voltage(
            state.ca.cl.temperature, activity_o2 ** 0.5 * activity_h2,
        )

    def reversible_voltage_vs_RHE(self, cell, state) -> float:
        return calculate_reversible_cell_voltage(
            state.ca.cl.temperature,
            GasModel.species_partial_pressure(state.ca.cl, 'o2') / STD_PRESSURE,
        )

    def activation_overpotential(self, cell, state, theta_PtO: float = 0) -> float:
        h2_pp = GasModel.species_partial_pressure(state.an.cl, 'h2')
        state.membrane.h2_permeation_flux = cell.membrane.hydrogen_permeation_flux(
            h2_pp,
            state.membrane.temperature,
            state.an.cl.pressure - state.ca.cl.pressure,
            cell.membrane.water_vol_fraction(state.membrane.water_content, state.mea_water_molar_volume),
        )
        state.crossover_current = state.membrane.h2_permeation_flux * (2 * FARADAY_CONSTANT)
        omega_PtO_voltage_drop = (
            cell.ca.cl.omega_PtO * theta_PtO
            / (cell.ca.cl.reaction.number_of_electrons * cell.ca.cl.reaction.charge_transfer_coeff * FARADAY_CONSTANT)
        )
        orr_overpotential = cell.ca.cl.reaction.tafel_overpotential(
            (state.current_density + state.crossover_current)
            / (cell.ca.cl.ecsa * cell.ca.cl.platinum_loading * (1 - theta_PtO)),
            state.ca.cl.temperature,
            GasModel.species_partial_pressure(state.ca.cl, 'o2'),
        )
        hor_overpotential = 0
        return orr_overpotential + omega_PtO_voltage_drop + hor_overpotential

    def high_frequency_resistance(self, cell, state) -> float:
        return (
            cell.membrane.proton_resistance(state.membrane.temperature, water_saturation=state.ca.cl.liquid_saturation)
            + cell.electrical_resistance
        )

    def ohmic_overpotential(self, cell, state) -> float:
        side_state = state.ca
        side_cell = cell.ca
        state.ca.cl.proton_resistance = 1. / (
            side_state.cl.non_wetting_saturation / side_cell.cl.effective_charge_resistance(
                state.current_density, side_state.liquid_eq_water_content, side_state.cl.temperature,
            ) + (1 - side_state.cl.non_wetting_saturation) / side_cell.cl.effective_charge_resistance(
                state.current_density, side_state.vapor_eq_water_content, side_state.cl.temperature,
            )
        )
        return state.current_density * (state.ca.cl.proton_resistance + self.high_frequency_resistance(cell, state))

    def calculate_theta_PtO(self, cell, state) -> float:
        E_rev_vs_RHE = self.reversible_voltage_vs_RHE(cell, state)
        theta_PtO = 0
        if cell.ca.cl.omega_PtO > 0:
            eps_max = 10
            eta_act = 0
            while eps_max > 0.001:
                eta_act_old = eta_act
                eta_act = self.activation_overpotential(cell, state, theta_PtO)
                theta_PtO = 0.5 * theta_PtO + 0.5 / (1 + np.exp(22.4 * (0.818 - E_rev_vs_RHE + eta_act)))
                eps_max = np.mean(np.abs(eta_act - eta_act_old))
        state.ca.cl.theta_catalyst = theta_PtO
        return theta_PtO

    def compute_cell_voltage(self, cell, state) -> float:
        """Compute cell voltage, writing E_rev, eta_act, eta_ohm and cell_voltage onto *state*."""
        theta_PtO = self.calculate_theta_PtO(cell, state)
        E_rev = self.reversible_cell_voltage(cell, state)
        eta_ohm = self.ohmic_overpotential(cell, state)
        eta_act = self.activation_overpotential(cell, state, theta_PtO)
        state.cell_voltage = np.maximum(0, E_rev - eta_act - eta_ohm)
        state.E_rev = E_rev
        state.eta_act = eta_act
        state.eta_ohm = eta_ohm
        return state.cell_voltage
