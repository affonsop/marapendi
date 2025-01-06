"""
Module providing a classes to model porous layers in electrochemical cells. 
"""
from dataclasses import dataclass, field
import numpy as np 
import cantera as ct

from .gas_composition import GasComposition, index_h2, index_o2, index_h2ov, species_indexes
from .tools import calculate_arrhenius_term
from .electrochemistry import ElectrochemicalReaction 
from .transport import PorousGasResistanceModel
from .water import water_saturation_pressure


@dataclass
class PorousLayer():
    thickness: float = 1e-3
    gas: GasComposition = field(default_factory=GasComposition)
    porosity: float = 1
    effective_gas_diffusion_ratio: float = 1
    pore_diameter: float=1e12
    transport_resistance_model: PorousGasResistanceModel = field(default_factory=PorousGasResistanceModel)
    water_saturation: float = 0
    thermal_conductivity: float = 1e12 

    def get_o2_mole_fraction(self):
        return self.gas.states.X[...,index_o2]
    
    def get_h2_mole_fraction(self):
        return self.gas.states.X[...,index_h2]
    
    def get_species_mole_fraction(self, species):
        return self.gas.states.X[...,species_indexes[species]] 
    
    def get_species_diffusion_coefficient(self, species): 
        return self.gas.states.mix_diff_coeffs_mole[...,species_indexes[species]]

    def get_species_molecular_weight(self, species): 
        return self.gas.states.molecular_weights[...,species_indexes[species]]

    def get_gas_temperature(self):
        return self.gas.temperature()

    def get_gas_pressure(self): 
        return self.gas.pressure()
    
    def get_species_concentrations(self, species): 
        return self.gas.states.concentrations[...,species_indexes[species]] 

    def get_species_partial_pressure(self, species): 
        return self.gas.states.X[...,species_indexes[species]] * self.gas.states.P

    def get_vapor_pressure(self): 
        return self.gas.states.X[...,index_h2ov] * self.gas.pressure()
     
    def get_relative_humidity(self):
        return self.gas.relative_humidity
    
    def get_saturation_pressure(self): 
        return self.gas.saturation_pressure
    
    def get_saturation_concentration(self): 
        return self.get_saturation_pressure() / (ct.gas_constant * self.get_gas_temperature()) 

    def get_vapor_concentration(self): 
        return self.get_vapor_pressure() / (ct.gas_constant * self.get_gas_temperature())
    
    def calculate_gas_transport_resistance(self, species='o2'): 
        return self.transport_resistance_model.total_diffusion_resistance(
            self, 
            self.get_gas_temperature(), 
            self.get_species_diffusion_coefficient(species), 
            self.get_species_molecular_weight(species), 
            self.water_saturation)
    
    def calculate_heat_transfer_resistance(self): 
        return self.thickness / self.thermal_conductivity
    
@dataclass 
class CatalystLayerIonomerModel: 
    density: float = 2004 
    equivalent_weight: float = 952. 
    hydrated_proton_conductivity: float = 11 # S/m
    proton_conductivity_exponent: float = 2.7

    def o2_film_resistance(self, water_content, temperature= 353.15):
        # Linear regression of data from Jinnouchi et al. (2021), neglecting bulk diffusion.
        # Activation energy obtained by Kudo et al. (2006).
        return (7957 - 408*water_content) * calculate_arrhenius_term(24e6, temperature, 353.15)

    def proton_conductivity(self, relative_humidity, water_content, temperature):
        return self.hydrated_proton_conductivity * relative_humidity ** (self.proton_conductivity_exponent) * calculate_arrhenius_term(11e6, temperature, 353.15) # Following measurements of Hutapea et al. (2023)

NafionD2020 = CatalystLayerIonomerModel(density=2004., equivalent_weight=952.)


@dataclass 
class CatalystLayer(PorousLayer):
    thickness: float = 10e-6
    porosity: float = 0
    ionomer: CatalystLayerIonomerModel = field(default_factory=CatalystLayerIonomerModel)
    reaction: ElectrochemicalReaction = field(default_factory=ElectrochemicalReaction)
    platinum_loading: float = 0.2e-6*1e4
    catalyst_platinum_weight_percent: float = 0.5
    ionomer_to_carbon_ratio: float = 0.75 
    platinum_density: float = 21450. 
    carbon_density: float = 1950.
    ecsa: float = 70e3 
    platinum_vol_surface_area: float = 0 
    carbon_agglomerate_radius: float = 20e-9 
    ionomer_vol_fraction: float = 0
    ionomer_film_thickness: float = 0 
    
    def __post_init__(self): 
        if self.platinum_vol_surface_area == 0: 
            self.platinum_vol_surface_area = self.platinum_loading * self.ecsa / self.thickness 
        if self.porosity == 0:
            self.carbon_loading  = self.platinum_loading * (1/self.catalyst_platinum_weight_percent - 1)
            self.carbon_vol_fraction = self.carbon_loading / self.thickness / self.carbon_density
            self.platinum_vol_fraction = self.platinum_loading / self.thickness / self.platinum_density
            self.catalyst_vol_fraction = self.platinum_vol_fraction + self.carbon_vol_fraction 
            self.ionomer_vol_fraction = self.carbon_loading / self.thickness * self.ionomer_to_carbon_ratio / self.ionomer.density 
            self.porosity = 1 - self.catalyst_vol_fraction - self.ionomer_vol_fraction
        else:
            self.catalyst_vol_fraction = 1 - self.ionomer_vol_fraction - self.porosity
        self.carbon_agglomerate_surface = 4 * np.pi * self.carbon_agglomerate_radius ** 2
        self.carbon_agglomerate_volume =  self.carbon_agglomerate_surface * self.carbon_agglomerate_radius / 3.
        self.carbon_agglomerate_number_density = self.catalyst_vol_fraction  / self.carbon_agglomerate_volume
        if self.ionomer_film_thickness == 0: 
            self.ionomer_film_thickness = self.ionomer_vol_fraction / (self.carbon_agglomerate_number_density * self.carbon_agglomerate_surface)
        self.ionomer_specific_surface = 4 * np.pi * (self.carbon_agglomerate_radius + self.ionomer_film_thickness) ** 2 * self.carbon_agglomerate_number_density
        self.effective_gas_diffusion_ratio = self.porosity ** 1.5 

    def calculate_o2_film_resistance(self, ionomer_water_content, temperature): 
        return self.ionomer.o2_film_resistance(ionomer_water_content, temperature) / self.ionomer_specific_surface / self.thickness
    
    def calculate_ionomer_sheet_proton_resistance(self, relative_humidity, ionomer_water_content, temperature): 
        # Consider ionomer_tortuosity = 1
        ionomer_proton_conductivity = self.ionomer.proton_conductivity(relative_humidity, ionomer_water_content, temperature)
        return self.thickness / (self.ionomer_vol_fraction * ionomer_proton_conductivity)
    
    def calculate_effective_proton_resistance(self,current_density, relative_humidity, ionomer_water_content, temperature): 
        # Based on the method proposed by Goshtasbi et al. (2020), based on Neyerlin et al. (2007)
        ionomer_sheet_resistance = self.calculate_ionomer_sheet_proton_resistance(relative_humidity, ionomer_water_content, temperature)
        nu = ionomer_sheet_resistance * current_density / self.reaction.tafel_slope(temperature)
        xi_neyerlin = nu * (-8.287e-3 * nu + 0.7184) - 2.072e-3
        return ionomer_sheet_resistance / (3 + xi_neyerlin)