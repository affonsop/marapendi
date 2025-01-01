"""
Module providing a fuel cell class intended to be the base class for different fuel cell models. 
"""
from dataclasses import dataclass, field
import numpy as np 

from .electrochemistry import ElectrochemicalReaction, calculate_reversible_cell_voltage
from .gas import GasComposition
from .membrane import Membrane

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

    def set_conditions(self, temperature=None, rh=None, pressure=None, dry_o2_mole_fraction=None, dry_h2_mole_fraction=None, stoichiometry=None): 
        if not temperature: self.temperature = temperature
        if not rh: self.rh = rh
        if not pressure: self.pressure = pressure
        if not dry_o2_mole_fraction: self.dry_o2_mole_fraction = dry_o2_mole_fraction
        if not dry_h2_mole_fraction: self.dry_h2_mole_fraction = dry_h2_mole_fraction
        if not stoichiometry: self.stoichiometry = stoichiometry
        self.__post_init__()
    


# @dataclass 
# class CellSide: 
#     channel_conditions: ChannelConditions

@dataclass 
class CellComponent: 
    temperature: float = 300.

@dataclass
class PorousLayer(CellComponent):
    gas: GasComposition = field(default_factory=GasComposition)
    
    def __post_init__(self): 
        self.gas.set_temperature(self.temperature)

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
class GasFlowChannel(CellComponent):
    gas: GasComposition = field(default_factory=GasComposition)
    reactant: int = 0
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
        
    def get_o2_mole_fraction(self):
        return self.gas.gas.X[0]
    
    def get_h2_mole_fraction(self):
        return self.gas.gas.X[2]

    def get_reactant_mole_fraction(self): 
        return self.gas.gas.X[self.reactant]
    
    def calculate_inlet_gas_flow_rate(self, reactant_consumption): 
        return self.inlet_stoichiometry * reactant_consumption / self.get_reactant_mole_fraction() * self.gas.gas.volume_mole
    
    def calculate_inlet_stochiometry(self, reactant_consumption): 
        return self.inlet_gas_flow_rate * self.get_reactant_mole_fraction() / self.gas.gas.volume_mole / reactant_consumption
    
    def get_o2_diffusion_coefficient(self): 
        return self.gas.gas.mix_diff_coeffs_mole[0]
    
    def get_transport_resistance(self, diffusion_coeff, volume_flow_rate): 
        return self.transport_resistance_model.total_resistance(self, diffusion_coeff, volume_flow_rate)



@dataclass
class FuelCellSide:
    cl: PorousLayer = field(default_factory=PorousLayer) 
    gdl: PorousLayer = field(default_factory=PorousLayer)
    mpl: PorousLayer = field(default_factory=PorousLayer)
    ch: GasFlowChannel = field(default_factory=GasFlowChannel)
    has_mpl: bool = False

    def __post_init__(self): 
        self.porous_layers = [self.cl, self.mpl, self.gdl] if self.has_mpl else [self.cl, self.gdl]
        self.components = self.porous_layers + [self.ch]

@dataclass
class FuelCell: 
    cell_area: float
    cell_number: int
    orr_reaction: ElectrochemicalReaction = field(default_factory=ElectrochemicalReaction)
    hor_reaction: ElectrochemicalReaction = field(default_factory=ElectrochemicalReaction)
    an: FuelCellSide = field(default_factory=FuelCellSide)
    ca: FuelCellSide = field(default_factory=FuelCellSide)
    membrane: Membrane = field(default_factory=Membrane)

    def reversible_cell_voltage(self, operating_conditions): 
        return calculate_reversible_cell_voltage(
            operating_conditions.temperature,
            operating_conditions.partial_pressure_o2,
            operating_conditions.partial_pressure_h2
        )
    
    def activation_overpotential(self, operating_conditions): 
        return self.orr_reaction.tafel_overpotential(
            operating_conditions.current_density,
            operating_conditions.temperature,
            operating_conditions.partial_pressure_o2
        )
    def ohmic_overpotential(self, operating_conditions): 
        return 0

    def cell_voltage(self, operating_conditions):

        reversible_cell_voltage = self.reversible_cell_voltage(operating_conditions)
        activation_overpotential_oer = self.activation_overpotential(operating_conditions)
        ohmic_overpotential = self.ohmic_overpotential(operating_conditions)
        
        return reversible_cell_voltage - activation_overpotential_oer