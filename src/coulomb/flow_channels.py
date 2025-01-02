"""
Module providing classes to model flow channels. 
"""
from dataclasses import dataclass, field
import numpy as np 
import cantera as ct

from .gas_composition import species_indexes
from .porous_layers import PorousLayer

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
class ChannelTransportResistanceModel: 
    A_ch: float = 1.0
    B_ch: float = 1.0

    def molecular_diffusion_resistance(self, channel, diffusion_coefficient): 
        return self.A_ch * channel.half_width / diffusion_coefficient
    
    def convection_resistance(self, channel, volume_flow_rate): 
        return self.B_ch * channel.length / channel.half_width * channel.total_flow_section / volume_flow_rate

    def total_resistance(self, channel, diffusion_coefficient, volume_flow_rate): 
        return (self.molecular_diffusion_resistance(channel, diffusion_coefficient) +
                self.convection_resistance(channel, volume_flow_rate)) 

@dataclass
class GasFlowChannel(PorousLayer):
    reactant: str = 'o2'
    inlet_stoichiometry: float = 0 
    inlet_gas_flow_rate: float = 0
    width: float = 1e-3
    height: float = 1e-3
    length: float = 100e-3
    n_parallel: int = 14
    transport_resistance_model: ChannelTransportResistanceModel = field(default_factory=ChannelTransportResistanceModel)

    def __post_init__(self): 
        self.gas.set_temperature(self.temperature)
        self.hydraulic_diameter = 2 * self.width * self.height / (self.width + self.height)
        self.channel_flow_section = self.width * self.height
        self.half_width = 0.5 * self.width
        self.total_flow_section = self.n_parallel * self.channel_flow_section
        
    def set_inlet_stoichiometry(self, stoichiometry):
        self.inlet_stoichiometry = stoichiometry

    def get_reactant_mole_fraction(self): 
        return self.gas.gas.X[species_indexes[self.reactant]]
    
   
    def calculate_inlet_gas_flow_rate(self, reactant_consumption): 
        return self.inlet_stoichiometry * reactant_consumption / self.get_reactant_mole_fraction() * self.gas.gas.volume_mole
    
    def calculate_inlet_stochiometry(self, reactant_consumption): 
        return self.inlet_gas_flow_rate * self.get_reactant_mole_fraction() / self.gas.gas.volume_mole / reactant_consumption

    def calculate_transport_resistance(self, diffusion_coeff, volume_flow_rate): 
        return self.transport_resistance_model.total_resistance(self, diffusion_coeff, volume_flow_rate)
