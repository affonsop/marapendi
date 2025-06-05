from dataclasses import dataclass, field
import numpy as np
from .water import water_saturation_pressure 

@dataclass
class ElectrolyteSolution(): 
    temperature: float = 298.15
    molality: float = 0 
    weight_percent: float = 0
    electrolyte_molecular_weight: float = 56.105

    def __post_init__(self): 
        if self.molality == 0:
            self.molality = self.calculate_molality(self.weight_percent)
        elif self.weight_percent == 0:
            self.weight_percent = self.calculate_weight_percent(self.molality)
        self.set_temperature(self.temperature)

    def set_temperature(self, temperature): 
        self.temperature = temperature
        self.density = self.calculate_density(self.weight_percent, temperature)
        self.molarity = self.calculate_molarity(self.weight_percent, temperature)
        self.ionic_conductivity = self.calculate_ionic_conductivity(self.molarity, self.weight_percent, self.temperature)
        self.water_sat_pressure = water_saturation_pressure(temperature)
        self.solution_sat_pressure = self.calculate_solution_saturation_pressure(self.molality, self.water_sat_pressure)

    def calculate_weight_percent(self, molality): 
        return 100. * self.electrolyte_molecular_weight * molality / (1 + self.electrolyte_molecular_weight * molality)

    def calculate_molality(self, weight_percent): 
        # In kmol/kg
        return weight_percent / self.electrolyte_molecular_weight / (100 - weight_percent)
    
    def calculate_molarity(self, weight_percent, temperature): 
        # In kmol/m3
        return weight_percent/100. * self.calculate_density(weight_percent, temperature) / self.electrolyte_molecular_weight
    
    def calculate_ionic_conductivity(self, molarity, weight_percent, temperature):
        return 1e-12

    def calculate_density(self, weight_percent, temperature): 
        return 1000. 
    
    def calculate_solution_saturation_pressure(self, molality, water_sat_pressure): 
        return water_sat_pressure

@dataclass
class KOH_solution(ElectrolyteSolution): 
    def __post_init__(self):
        super().__post_init__() 
        self.electrolyte_molecular_weight = 56.105
    


    def calculate_density(self, weight_percent, temperature): 
        # Eq. 4 in Hodges et al. (2023)
        T = temperature - 273.15
        return (5.1998e-6 * T ** 3 -  
         39.771334e-4 * T ** 2 -
         848.089182e-4 * T + 
         1001.5409980109) * np.exp(0.0086 * weight_percent)

    def calculate_ionic_conductivity(self, molarity, weight_percent, temperature):  
        # Eq. 5 and 6 in Hodges et al. (2023)
        low_temperature_conductivity = molarity * (-2.041 - 0.0028 * molarity +
                                                    0.005332 * temperature + 
                                                    207.2 / temperature + 
                                                    0.001043 * molarity ** 2 -
                                                    3e-7 * molarity  * temperature ** 2)
        high_temperature_conductivity =  weight_percent * (2.2204e-3 -
                                                           1.3077e-3 * weight_percent + 
                                                           3.3647 * temperature - 
                                                           10.7021 / temperature + 
                                                           7.0101e-6 * weight_percent ** 2 - 
                                                           3.2033e-9 * weight_percent * temperature ** 2)
    
        return np.where(temperature < 353.15, 
                        low_temperature_conductivity, 
                        high_temperature_conductivity)
                 

    def calculate_solution_saturation_pressure(self, molality, water_sat_pressure):
        # Eq. 6 in Balej (1985)
        molality_mol_per_kg = molality * 1000.
        log_p_sat = np.log10(water_sat_pressure/1e5)
        return 10 ** (5 + log_p_sat - molality_mol_per_kg  * (
                (0.01508 + 0.0012062 * log_p_sat) + 
                (0.0016788 - 5.6024e-4 * log_p_sat) * molality_mol_per_kg -
                (2.25887e-5 - 7.8228e-6 * log_p_sat) * molality_mol_per_kg ** 2))
    
KOH_1M = KOH_solution(temperature=298.15,weight_percent=5.3732)
KOH_2M = KOH_solution(temperature=298.15,weight_percent=10.3)
KOH_5M = KOH_solution(temperature=298.15,weight_percent=23.072)
KOH_20_wt_percent = KOH_solution(temperature=298.15, weight_percent=20.)
KOH_45_wt_percent = KOH_solution(temperature=298.15, weight_percent=45.)