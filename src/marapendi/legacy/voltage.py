"""
Voltage model for the vendored ``legacy`` fuel-cell physics.

Extracts the voltage-related calculations of
:class:`marapendi.legacy.fuelcell.FuelCell` (reversible voltage, activation
overpotential, ohmic overpotential, high-frequency resistance, and the
resulting cell voltage) into a separate model that operates on a
:class:`marapendi.legacy.state.FuelCellState`.

Each method takes a single ``state`` argument (created via
``fuel_cell.to_state()``) and writes its results back into it. The dynamic
inputs (gas partial pressures, water content, saturations, ...) that are not
yet captured as plain data in ``FuelCellState`` are read via
``state.fuel_cell``, the reference to the ``FuelCell`` the state was
snapshotted from.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cantera as ct

from .electrochemistry import calculate_reversible_cell_voltage, STD_PRESSURE
from .state import FuelCellState


@dataclass
class VoltageModel:
    """Stateless collection of voltage-related calculations."""

    def reversible_cell_voltage(self, state: FuelCellState):
        fuel_cell = state.fuel_cell
        activity_o2 = fuel_cell.ca.cl.species_partial_pressure('o2') / STD_PRESSURE
        activity_h2 = fuel_cell.an.cl.species_partial_pressure('h2') / STD_PRESSURE
        return calculate_reversible_cell_voltage(
            fuel_cell.ca.cl.temperature,
            activity_o2 ** 0.5 * activity_h2,
        )

    def reversible_voltage_vs_RHE(self, state: FuelCellState):
        fuel_cell = state.fuel_cell
        return calculate_reversible_cell_voltage(
            fuel_cell.ca.cl.temperature,
            fuel_cell.ca.cl.species_partial_pressure('o2') / STD_PRESSURE,
        )

    def calculate_h2_permeation_flux(self, state: FuelCellState):
        fuel_cell = state.fuel_cell
        state.membrane.h2_permeation_flux = fuel_cell.membrane.hydrogen_permeation_flux(
            fuel_cell.an.cl.species_partial_pressure('h2'),
            fuel_cell.membrane.temperature,
            fuel_cell.an.cl.pressure - fuel_cell.ca.cl.pressure,
            fuel_cell.membrane.water_vol_fraction(
                fuel_cell.membrane.water_content, fuel_cell.mea_water_molar_volume
            ),
        )
        return state.membrane.h2_permeation_flux

    def activation_overpotential(self, state: FuelCellState, theta_PtO=0):
        fuel_cell = state.fuel_cell
        crossover_current = state.membrane.h2_permeation_flux * (2 * ct.faraday)
        state.ca.cl.crossover_current = crossover_current

        omega_PtO_voltage_drop = (
            fuel_cell.ca.cl.omega_PtO * theta_PtO
            / (fuel_cell.ca.cl.reaction.number_of_electrons
               * fuel_cell.ca.cl.reaction.charge_transfer_coeff * ct.faraday)
        )
        state.ca.cl.overpotential = fuel_cell.ca.cl.reaction.tafel_overpotential(
            (state.current_density + crossover_current)
            / (fuel_cell.ca.cl.ecsa * fuel_cell.ca.cl.platinum_loading * (1 - theta_PtO)),
            fuel_cell.ca.cl.temperature,
            fuel_cell.ca.cl.species_partial_pressure('o2'),
        )
        state.an.cl.overpotential = 0
        return state.ca.cl.overpotential + omega_PtO_voltage_drop + state.an.cl.overpotential

    def high_frequency_resistance(self, state: FuelCellState):
        fuel_cell = state.fuel_cell
        return (
            fuel_cell.membrane.proton_resistance(
                fuel_cell.membrane.temperature,
                water_saturation=fuel_cell.ca.cl.liquid_saturation,
            )
            + fuel_cell.electrical_resistance
        )

    def ohmic_overpotential(self, state: FuelCellState):
        fuel_cell = state.fuel_cell
        side, side_state = fuel_cell.ca, state.ca
        side_state.cl.proton_resistance = 1. / (
            side.cl.non_wetting_saturation / side.cl.effective_charge_resistance(
                state.current_density, side.liquid_eq_water_content, side.cl.temperature
            ) + (1 - side.cl.non_wetting_saturation) / side.cl.effective_charge_resistance(
                state.current_density, side.vapor_eq_water_content, side.cl.temperature
            )
        )
        hfr = self.high_frequency_resistance(state)
        state.membrane.proton_resistance = hfr - fuel_cell.electrical_resistance
        return state.current_density * (side_state.cl.proton_resistance + hfr)

    def calculate_theta_PtO(self, state: FuelCellState):
        fuel_cell = state.fuel_cell
        E_rev_vs_RHE = self.reversible_voltage_vs_RHE(state)
        theta_PtO = 0
        if fuel_cell.ca.cl.omega_PtO > 0:
            eps_max = 10
            eta_act = 0
            while eps_max > 0.001:
                eta_act_old = eta_act
                eta_act = self.activation_overpotential(state, theta_PtO)
                theta_PtO = 0.5 * theta_PtO + 0.5 / (1 + np.exp(22.4 * (0.818 - E_rev_vs_RHE + eta_act)))
                eps_max = np.mean(np.abs(eta_act - eta_act_old))
        state.ca.cl.theta_catalyst = theta_PtO
        return theta_PtO

    def compute_cell_voltage(self, state: FuelCellState) -> FuelCellState:
        """Compute the cell voltage for ``state``, writing results back into it.

        ``state`` must be a snapshot obtained via ``fuel_cell.to_state()``.
        """
        self.calculate_h2_permeation_flux(state)
        theta_PtO = self.calculate_theta_PtO(state)
        state.E_rev = self.reversible_cell_voltage(state)
        state.eta_ohm = self.ohmic_overpotential(state)
        state.eta_act = self.activation_overpotential(state, theta_PtO)
        state.cell_voltage = np.maximum(0, state.E_rev - state.eta_act - state.eta_ohm)
        return state
