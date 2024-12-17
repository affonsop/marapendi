from dataclasses import dataclass
import numpy as np 

from .electrochemistry import ElectrochemicalReaction, calculate_reversible_cell_voltage, calculate_tafel_overpotential, calculate_exchange_current_density

@dataclass

class FuelCell: 
    cell_area: float
    cell_number: int
    orr_reaction: ElectrochemicalReaction
    hor_reaction: ElectrochemicalReaction


    def cell_voltage(self, operating_conditions):

        reversible_cell_voltage = calculate_reversible_cell_voltage(
            operating_conditions.temperature,
            operating_conditions.partial_pressure_o2,
            operating_conditions.partial_pressure_h2
        )

        activation_potential_oer = self.orr_reaction.tafel_overpotential(
            operating_conditions.current_density,
            operating_conditions.temperature,
            operating_conditions.partial_pressure_o2)
        print(reversible_cell_voltage, reversible_cell_voltage - activation_potential_oer,activation_potential_oer, operating_conditions.current_density)
        return reversible_cell_voltage - activation_potential_oer