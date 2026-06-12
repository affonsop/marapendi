"""
Voltage model: reversible voltage, overpotentials and resulting cell voltage.

:class:`VoltageModel` computes the cell voltage of a
:class:`~marapendi.cell.Cell` for a given :class:`~marapendi.state.CellState`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .electrochemistry import calculate_reversible_cell_voltage, STD_PRESSURE
from .water import water_molar_volume
from .cell import Cell
from .constants import FARADAY_CONSTANT
from .gas import GasModel
from .state import CellState


@dataclass
class VoltageModel:
    """Stateless collection of voltage-related calculations."""

    def reversible_cell_voltage(self, cell: Cell, state: CellState) -> float:
        activity_o2 = GasModel.species_partial_pressure(state.ca.cl, 'o2') / STD_PRESSURE
        activity_h2 = GasModel.species_partial_pressure(state.an.cl, 'h2') / STD_PRESSURE
        return calculate_reversible_cell_voltage(state.ca.cl.temperature, activity_o2 ** 0.5 * activity_h2)

    def reversible_voltage_vs_RHE(self, cell: Cell, state: CellState) -> float:
        return calculate_reversible_cell_voltage(
            state.ca.cl.temperature,
            GasModel.species_partial_pressure(state.ca.cl, 'o2') / STD_PRESSURE,
        )

    def calculate_h2_permeation_flux(self, cell: Cell, state: CellState) -> float:
        state.membrane.h2_permeation_flux = cell.membrane.hydrogen_permeation_flux(
            GasModel.species_partial_pressure(state.an.cl, 'h2'),
            state.membrane.temperature,
            state.an.cl.pressure - state.ca.cl.pressure,
            cell.membrane.water_vol_fraction(
                state.membrane.water_content, water_molar_volume(state.membrane.temperature),
            ),
        )
        return state.membrane.h2_permeation_flux

    def activation_overpotential(self, cell: Cell, state: CellState, theta_PtO: float = 0) -> float:
        crossover_current = state.membrane.h2_permeation_flux * (2 * FARADAY_CONSTANT)
        state.ca.cl.crossover_current = crossover_current

        omega_PtO_voltage_drop = (
            cell.ca.cl.omega_PtO * theta_PtO
            / (cell.ca.cl.reaction.number_of_electrons * cell.ca.cl.reaction.charge_transfer_coeff * FARADAY_CONSTANT)
        )
        state.ca.cl.overpotential = cell.ca.cl.reaction.tafel_overpotential(
            (state.current_density + crossover_current)
            / (cell.ca.cl.ecsa * cell.ca.cl.platinum_loading * (1 - theta_PtO)),
            state.ca.cl.temperature,
            GasModel.species_partial_pressure(state.ca.cl, 'o2'),
        )
        state.an.cl.overpotential = 0
        return state.ca.cl.overpotential + omega_PtO_voltage_drop + state.an.cl.overpotential

    def high_frequency_resistance(self, cell: Cell, state: CellState) -> float:
        membrane_state = state.membrane
        liquid_equilibrated_conductivity = cell.membrane.proton_conductivity(
            membrane_state.liquid_eq_sat_water_profile, membrane_state.temperature,
        )
        vapor_equilibrated_conductivity = cell.membrane.proton_conductivity(
            membrane_state.vapor_eq_sat_water_profile, membrane_state.temperature,
        )
        water_saturation = state.ca.cl.non_wetting_saturation
        average_conductivity = (
            (1 - water_saturation) * vapor_equilibrated_conductivity
            + water_saturation * liquid_equilibrated_conductivity
        )
        return cell.membrane.dry_thickness / average_conductivity + cell.electrical_resistance

    def ohmic_overpotential(self, cell: Cell, state: CellState) -> float:
        side, side_state = cell.ca, state.ca
        non_wetting_saturation = side_state.cl.non_wetting_saturation
        side_state.cl.proton_resistance = 1. / (
            non_wetting_saturation / side.cl.effective_charge_resistance(
                state.current_density, side_state.liquid_eq_water_content, side_state.cl.temperature,
            ) + (1 - non_wetting_saturation) / side.cl.effective_charge_resistance(
                state.current_density, side_state.vapor_eq_water_content, side_state.cl.temperature,
            )
        )
        hfr = self.high_frequency_resistance(cell, state)
        state.membrane.proton_resistance = hfr - cell.electrical_resistance
        return state.current_density * (side_state.cl.proton_resistance + hfr)

    def calculate_theta_PtO(self, cell: Cell, state: CellState) -> float:
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

    def compute_cell_voltage(self, cell: Cell, state: CellState) -> CellState:
        """Compute the cell voltage for ``state``, writing results back into it."""
        self.calculate_h2_permeation_flux(cell, state)
        theta_PtO = self.calculate_theta_PtO(cell, state)
        state.E_rev = self.reversible_cell_voltage(cell, state)
        state.eta_ohm = self.ohmic_overpotential(cell, state)
        state.eta_act = self.activation_overpotential(cell, state, theta_PtO)
        state.cell_voltage = np.maximum(0, state.E_rev - state.eta_act - state.eta_ohm)
        return state
