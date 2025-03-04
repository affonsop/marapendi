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
from .water import water_kinematic_viscosity, water_surface_tension, water_molecular_weight


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
    absolute_permeability: float = 10000
    contact_angle: float = 120. 

    def __post_init__(self): 
        self.temperature = self.gas.temperature
        self.RT = ct.gas_constant * self.temperature
        self.pressure = self.gas.pressure 
        
        self.sqrt_abs_permeability_porosity = np.sqrt(self.absolute_permeability * self.porosity)
        self.cosinus_contact_angle = np.abs(np.cos(np.pi / 180 * self.contact_angle))

    def get_o2_mole_fraction(self):
        return self.gas.X[...,index_o2]
    
    def get_h2_mole_fraction(self):
        return self.gas.X[...,index_h2]
    
    def get_species_mole_fraction(self, species):
        return self.gas.X[...,species_indexes[species]] 
    
    def get_species_diffusion_coefficient(self, species): 
        return self.gas.species_diffusion_coefficient(species)
    
    def get_species_molecular_weight(self, species): 
        return self.gas.molecular_weights[species_indexes[species]]

    def get_gas_temperature(self):
        return self.temperature

    def get_gas_pressure(self): 
        return self.pressure
    
    def set_gas_composition(self, dry_o2_mole_fraction, dry_h2_mole_fraction, relative_humidity): 
        self.gas.set_composition(dry_o2_mole_fraction, dry_h2_mole_fraction, relative_humidity)

    def set_gas_pressure(self, pressure):
        self.pressure = pressure
        self.gas.set_pressure(pressure)

    def set_gas_temperature(self, temperature):
        self.temperature = temperature 
        self.RT = ct.gas_constant * self.temperature
        self.gas.set_temperature(temperature)
        
    def set_gas_temperature_and_pressure(self, temperature, pressure):
        self.temperature = temperature 
        self.RT = ct.gas_constant * self.temperature
        self.pressure = pressure
        self.gas.set_temperature_and_pressure(temperature, pressure)

    def get_species_concentrations(self, species): 
        return self.gas.X[...,species_indexes[species]] * self.gas.pressure / self.RT

    def get_species_partial_pressure(self, species): 
        return self.gas.X[...,species_indexes[species]] * self.pressure

    def get_vapor_pressure(self): 
        return self.gas.X[...,index_h2ov] * self.pressure
     
    def get_relative_humidity(self):
        return self.gas.relative_humidity
    
    def get_saturation_pressure(self): 
        return self.gas.saturation_pressure
    
    def get_saturation_concentration(self): 
        return self.get_saturation_pressure() / self.RT
    
    def get_vapor_concentration(self): 
        return self.get_vapor_pressure() / self.RT
    
    def calculate_gas_transport_resistance(self, species='o2'): 
        return self.transport_resistance_model.total_diffusion_resistance(
            self, 
            self.temperature, 
            self.get_species_diffusion_coefficient(species), 
            self.get_species_molecular_weight(species), 
            self.water_saturation)
    
    def calculate_heat_transfer_resistance(self): 
        return self.thickness / self.thermal_conductivity
    
    def calculate_saturation_absolute_flow_resistance(self): 
        return ((self.thickness / self.sqrt_abs_permeability_porosity) *
                water_kinematic_viscosity(self.temperature) * water_molecular_weight /
                (self.cosinus_contact_angle * water_surface_tension(self.temperature)))
        

@dataclass 
class CatalystLayerIonomerModel: 
    dry_density: float = 2004 
    equivalent_weight: float = 952. 
    
    hydrated_proton_conductivity: float = 11 # S/m
    proton_conductivity_water_content_exponent: float = 0
    proton_conductivity_rh_exponent: float = 2.7
    proton_conductivity_activation_energy: float = 11e6
    hydrated_o2_diffusion: float = 1.14698e-10*14**0.708
    o2_diffusion_exponent: float = 0.708
    o2_diffusion_activation_energy: float = 24e6

    def __post_init__(self): 
        self.dry_concentration = self.dry_density / self.equivalent_weight 

    def o2_film_resistance(self, water_content, temperature= 353.15):
        # Linear regression of data from Jinnouchi et al. (2021), neglecting bulk diffusion.
        # Activation energy obtained by Kudo et al. (2006).
        return (self.hydrated_o2_diffusion * 
                (water_content/14) ** self.o2_diffusion_exponent *
                calculate_arrhenius_term(self.o2_diffusion_activation_energy, temperature, 353.15)) 

    def proton_conductivity(self, relative_humidity, water_content, temperature):
        return (self.hydrated_proton_conductivity *
                (water_content/14.) ** (self.proton_conductivity_water_content_exponent) *
                relative_humidity ** self.proton_conductivity_rh_exponent *
                calculate_arrhenius_term(self.proton_conductivity_activation_energy, temperature, 353.15)) # Following measurements of Hutapea et al. (2023)

NafionD2020 = CatalystLayerIonomerModel(dry_density=2004., equivalent_weight=952.)


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
    contact_angle: float = 95.
    omega_PtO: float = 3000e3

    def __post_init__(self):

        

        
        if self.platinum_vol_surface_area == 0: 
            self.platinum_vol_surface_area = self.platinum_loading * self.ecsa / self.thickness 
        if self.porosity == 0:
            self.carbon_loading  = self.platinum_loading * (1/self.catalyst_platinum_weight_percent - 1)
            self.carbon_vol_fraction = self.carbon_loading / self.thickness / self.carbon_density
            self.platinum_vol_fraction = self.platinum_loading / self.thickness / self.platinum_density
            self.catalyst_vol_fraction = self.platinum_vol_fraction + self.carbon_vol_fraction 
            self.ionomer_vol_fraction = self.carbon_loading / self.thickness * self.ionomer_to_carbon_ratio / self.ionomer.dry_density 
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
        PorousLayer.__post_init__(self)

    def calculate_o2_film_resistance(self, ionomer_water_content, temperature): 
        return  self.ionomer_film_thickness / self.ionomer.o2_film_resistance(ionomer_water_content, temperature) # / self.ionomer_specific_surface / self.thickness
    
    def calculate_ionomer_sheet_proton_resistance(self, relative_humidity, ionomer_water_content, temperature): 
        # Consider ionomer_tortuosity = 1
        ionomer_proton_conductivity = self.ionomer.proton_conductivity(relative_humidity, ionomer_water_content, temperature)
        return self.thickness / (self.ionomer_vol_fraction * ionomer_proton_conductivity)
    
    def calculate_effective_proton_resistance(self,current_density, relative_humidity, ionomer_water_content, temperature): 
        # Based on the method proposed by Goshtasbi et al. (2020), based on Neyerlin et al. (2007)
        self.ionomer_sheet_resistance = self.calculate_ionomer_sheet_proton_resistance(relative_humidity, ionomer_water_content, temperature)
        nu = np.minimum(self.ionomer_sheet_resistance * current_density / self.reaction.tafel_slope(temperature), 10) # Goshtasbi parametrization valid for nu < 10
        self.xi_neyerlin = nu * (-8.287e-3 * nu + 0.7184) - 2.072e-3
        return self.ionomer_sheet_resistance / (3 + self.xi_neyerlin)