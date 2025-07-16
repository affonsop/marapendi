"""
Module providing a classes to model ionomers in catalyst layers. 
"""
from dataclasses import dataclass, field
import numpy as np 
import cantera as ct

from .tools import calculate_arrhenius_term
from .water import water_molecular_weight, water_molar_volume, water_density
from .membrane import Membrane

@dataclass 
class CatalystLayerIonomer(Membrane): 
    dry_density: float = 2004 
    equivalent_weight: float = 952. 
    
    def __post_init__(self): 
        Membrane.__post_init__(self)

    def wet_density(self, water_content, temperature):
        water_mass = water_molecular_weight * water_content
        return self.equivalent_weight + water_mass / (self.equivalent_weight / self.dry_density + water_mass / water_density(temperature))

    def wet_expansion_factor(self, water_content, temperature):
        water_mass = water_molecular_weight * water_content 
        return 1 + self.dry_density * water_mass / self.equivalent_weight / water_density(temperature)

    def charge_conductivity(self, water_content, temperature, charge='proton'): 
        if charge == 'proton':
            return self.proton_conductivity(water_content, temperature)
        elif charge == 'hydroxide': 
            return self.hydroxide_conductivity(water_content, temperature)
    
    def charge_resistance(self, water_content, temperature, charge='proton'): 
        return self.dry_thickness / self.charge_conductivity(water_content, temperature, charge)

@dataclass
class PFSAIonomer(CatalystLayerIonomer): 
    conductivity_correction: float = 1
    conductivity_exp: float = 1.5
    hydrated_proton_conductivity: float = 11 # S/m
    conductivity_activation_energy: float = 11e6
    hydrated_o2_diffusion: float = 1.14698e-10*14**0.708
    o2_diffusion_exponent: float = 0.708
    o2_diffusion_activation_energy: float = 24e6


    def o2_film_diffusion_coefficient(self, water_content, temperature= 353.15):
        # Linear regression of data from Jinnouchi et al. (2021), neglecting bulk diffusion.
        # Activation energy obtained by Kudo et al. (2006).
        return (self.hydrated_o2_diffusion * 
                (water_content/14) ** self.o2_diffusion_exponent *
                calculate_arrhenius_term(self.o2_diffusion_activation_energy, temperature, 353.15)) 

    def o2_permeability(self, water_content, temperature= 353.15):
        fv = self.water_vol_fraction(water_content, water_molar_volume(temperature))
        RT = ct.gas_constant * temperature
        return (6.74e-12 * np.exp(-21280e3/RT) + fv * 50.5e-12 * np.exp(-20470e3/RT)) * 1e-3

    def proton_conductivity(self, water_content, temperature):
        fv = self.water_vol_fraction(water_content ,water_molar_volume(temperature))
        return self.conductivity_correction * 50 * (np.maximum(fv, 0.11) - 0.1 ) ** self.conductivity_exp * calculate_arrhenius_term(self.conductivity_activation_energy, temperature, 298.15)
    
    def equilibrium_water_content(self, rh):
        return 21.669 * rh ** 3 - 27.692* rh **2 + 17.624 * rh + 0.688 # Fit from Jinnouchi et al. (2021), sup. material

NafionD2020 = PFSAIonomer(dry_density=2004., equivalent_weight=952.)

@dataclass
class PAPIonomer(CatalystLayerIonomer):
    dry_density: float = 1220.
    equivalent_weight: float = 1000/2.35
    
    def hydroxide_conductivity(self, water_content, temperature): 
        return np.interp(temperature, [298.15, 313.15, 333.15, 353.15], [6.63,	8.45,	11.44,	14.87])
    
 

