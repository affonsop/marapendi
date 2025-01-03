"""
Module providing a fuel cell class intended to be the base class for different fuel cell models. 
"""
from dataclasses import dataclass, field
import numpy as np 
import cantera as ct

from .electrochemistry import calculate_reversible_cell_voltage
from .porous_layers import PorousLayer, CatalystLayer
from .flow_channels import GasFlowChannel
from .membrane import Membrane

@dataclass
class FuelCellSide:
    cl: PorousLayer = field(default_factory=CatalystLayer) 
    gdl: PorousLayer = field(default_factory=PorousLayer)
    mpl: PorousLayer = field(default_factory=PorousLayer)
    ch: GasFlowChannel = field(default_factory=GasFlowChannel)
    has_mpl: bool = False

    def __post_init__(self): 
        self.porous_layers = [self.cl, self.mpl, self.gdl] if self.has_mpl else [self.cl, self.gdl]
        self.components = self.porous_layers + [self.ch]

    def set_catalyst_layer(self,cl): 
        self.cl = cl 
        self.__post_init__()
    
    def set_gas_diffusion_layer(self, gdl): 
        self.gdl = gdl
        self.__post_init__()
    
    def set_channel(self, ch): 
        self.ch = ch 
        self.__post_init__()
        
@dataclass
class FuelCell: 
    cell_area: float
    cell_number: int
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
        return self.ca.cl.reaction.tafel_overpotential(
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