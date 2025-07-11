"""
Module providing classes to model flow channels. 
"""
from dataclasses import dataclass, field
import numpy as np 
import cantera as ct

from .gas_composition import species_indexes
from .porous_layers import PorousLayer
from .transport_models import ChannelGasResistanceModel

@dataclass 
class ChannelConditions:  
    temperature: float
    rh: float
    pressure: float
    dry_o2_mole_fraction: float
    dry_h2_mole_fraction: float
    stoichiometry: float

    def __post_init__(self):
        pass

    def set_conditions(self, temperature=None,
                       rh=None, pressure=None,
                       dry_o2_mole_fraction=None,
                       dry_h2_mole_fraction=None,
                       stoichiometry=None):
        if not temperature:
            self.temperature = temperature
        if not rh:
            self.rh = rh
        if not pressure:
            self.pressure = pressure
        if not dry_o2_mole_fraction:
            self.dry_o2_mole_fraction = dry_o2_mole_fraction
        if not dry_h2_mole_fraction:
            self.dry_h2_mole_fraction = dry_h2_mole_fraction
        if not stoichiometry:
            self.stoichiometry = stoichiometry
        self.__post_init__()

@dataclass
class FlowChannel(PorousLayer):
    reactant: str = 'o2'
    inlet_stoichiometry: float = 0 
    inlet_gas_flow_rate: float = 1e-12
    inlet_liquid_flow_rate: float = 0
    inlet_liquid_saturation: float = 0
    width: float = 1e-3
    height: float = 1e-3
    length: float = 100e-3
    channel_land_ratio: float = 1.
    n_parallel: int = 14
    transport_resistance_model: ChannelGasResistanceModel = field(default_factory=ChannelGasResistanceModel)

    def __post_init__(self): 
        self.temperature = self.gas.temperature 
        self.pressure = self.gas.pressure 
        self.RT = ct.gas_constant * self.temperature
        self.hydraulic_diameter = 2 * self.width * self.height / (self.width + self.height)
        self.channel_flow_section = self.width * self.height
        self.half_width = 0.5 * self.width
        self.total_flow_section = self.n_parallel * self.channel_flow_section
        PorousLayer.__post_init__(self)

    def set_inlet_stoichiometry(self, stoichiometry):
        self.inlet_stoichiometry = stoichiometry

    def reactant_mole_fraction(self): 
        return self.gas.X[...,species_indexes[self.reactant]]
    
    def set_fixed_inlet_gas_flow_rate(self, inlet_gas_flow_rate): 
        self.inlet_gas_flow_rate = inlet_gas_flow_rate 
        self.inlet_liquid_saturation =  self.inlet_liquid_flow_rate / np.maximum(self.inlet_liquid_flow_rate +
                                                                        self.inlet_gas_flow_rate, 1e-12)
        
    def set_fixed_inlet_liquid_flow_rate(self, inlet_liquid_flow_rate): 
        self.inlet_liquid_flow_rate = inlet_liquid_flow_rate
        self.inlet_liquid_saturation =  self.inlet_liquid_flow_rate / np.maximum(self.inlet_liquid_flow_rate +
                                                                        self.inlet_gas_flow_rate, 1e-12)

    def set_inlet_gas_flow_rate_from_stoichiometry(self, reactant_consumption, stoichiometry=0, fixed_inlet_gas_flow_rate=0):
        try: 
            if stoichiometry > 0: 
                self.inlet_stoichiometry = stoichiometry 
        except ValueError: 
            self.inlet_stoichiometry = stoichiometry 
        inlet_gas_flow_rate = self.calculate_inlet_gas_flow_rate(reactant_consumption) + fixed_inlet_gas_flow_rate
        self.set_fixed_inlet_gas_flow_rate(inlet_gas_flow_rate)

   
    def calculate_inlet_gas_flow_rate(self, reactant_consumption): 
        return self.inlet_stoichiometry * reactant_consumption / self.reactant_mole_fraction() / self.gas.concentration()
    
    def calculate_inlet_stochiometry(self, reactant_consumption): 
        return self.inlet_gas_flow_rate * self.reactant_mole_fraction() * self.gas.concentration() / reactant_consumption

    def gas_transport_resistance(self, species, volume_flow_rate=None): 
        diffusion_coeff = self.species_diffusion_coefficient(species)
        return self.transport_resistance_model.total_resistance(self, diffusion_coeff, 
                                                                volume_flow_rate if volume_flow_rate else self.inlet_gas_flow_rate)
