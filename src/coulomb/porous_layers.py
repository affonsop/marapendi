"""
Module providing a classes to model porous layers in electrochemical cells. 
"""
from dataclasses import dataclass, field
import numpy as np 
import cantera as ct

from .gas_composition import GasComposition, index_o2, species_indexes

@dataclass 
class CellComponent: 
    temperature: float = 300.

@dataclass
class GasTransportResistanceModel:
    water_saturation_exponent: float = 3.0

    def water_saturation_correction(self, water_saturation):
        return (1 - water_saturation) ** self.water_saturation_exponent
    
    def molecular_diffusion_effective_length(self, layer, water_saturation=0):
        return layer.thickness / layer.effective_gas_diffusion_ratio / self.water_saturation_correction(water_saturation)
    
    def molecular_diffusion_resistance(self, layer, diffusion_coefficient, water_saturation=0):
        return self.molecular_diffusion_effective_length(layer, water_saturation) / diffusion_coefficient
    
    def knudsen_diffusivity(self,layer, temperature, molecular_weight):
        return layer.pore_diameter / 3 * np.sqrt(8 * ct.gas_constant * temperature / molecular_weight / np.pi)
    
    def total_diffusion_resistance(self, layer, temperature, diffusion_coefficient, molecular_weight, water_saturation=0):
        return self.molecular_diffusion_resistance(layer, diffusion_coefficient, water_saturation) + layer.thickness / self.knudsen_diffusivity(layer, temperature, molecular_weight)

@dataclass
class PorousLayer(CellComponent):
    thickness: float = 1e-3
    gas: GasComposition = field(default_factory=GasComposition)
    effective_gas_diffusion_ratio: float = 1
    pore_diameter: float=1e12
    transport_resistance_model: GasTransportResistanceModel = field(default_factory=GasTransportResistanceModel)

    def __post_init__(self):
        self.gas.set_temperature(self.temperature)

    def get_o2_mole_fraction(self):
        return self.gas.gas.X[index_o2]
    
    def get_h2_mole_fraction(self):
        return self.gas.gas.X[inedx_h2]
    
    def get_species_mole_fraction(self, species):
        return self.gas.gas.X[species_indexes[species]] 
    
    def get_species_diffusion_coefficient(self, species): 
        return self.gas.gas.mix_diff_coeffs_mole[species_indexes[species]]

    def get_species_molecular_weight(self, species): 
        return self.gas.gas.molecular_weights[species_indexes[species]]

    def get_gas_temperature(self): 
        return self.gas.gas.T

    def calculate_transport_resistance(self, species='o2'): 
        return self.transport_resistance_model.total_diffusion_resistance(
            self, 
            self.get_gas_temperature(), 
            self.get_species_diffusion_coefficient(species), 
            self.get_species_molecular_weight(species), 
            water_saturation=0)
    
