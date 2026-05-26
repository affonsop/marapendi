

import numpy as np 
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Dict, Any, List
from .membrane import Membrane
from .tools import Updatable 
from .porous_layers import PorousLayer
from .flow_channels import FlowChannel

@dataclass
class Layer(Updatable): 
    """
    A class representing one layer of an AEM/PEM cell.
    """
    name: str
    thickness: float
    thermal_conductivity: float
    ionomer_vol_fraction: float

@dataclass
class CellSide(Updatable):
    """
    A class representing one side (anode or cathode) of an AEM/PEM cell.
    """
    name: str = "side"
    cl: PorousLayer = field(default_factory=PorousLayer) 
    gdl: PorousLayer = field(default_factory=PorousLayer)
    mpl: PorousLayer = field(default_factory=PorousLayer)
    ch: FlowChannel = field(default_factory=FlowChannel)
    has_mpl: bool = False
    has_gdl: bool = True
  
    def __post_init__(self): 
        self.porous_layers = ([self.cl, self.mpl] if self.has_mpl else [self.cl]) + ([self.gdl] if self.has_gdl else [])
        self.layers = self.porous_layers + [self.ch]
    
@dataclass
class Cell(Updatable): 
    """
    A class representing an AEM/PEM cell.
    """
    name: str = "cell"
    ca: CellSide = field(default_factory=CellSide)
    an: CellSide = field(default_factory=CellSide)
    memb: Membrane = field(default_factory=Membrane)
    arrays: Dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self): 
        self.porous_layers = self.an.porous_layers[::-1] + self.ca.porous_layers 
        self.layers = self.an.layers[::-1] + [self.memb] + self.ca.layers
        self.build_property_arrays()

    def get_property_array(self, property_name: str):
        """Get array of properties from layers"""
        return np.array([getattr(layer, property_name) for layer in self.layers])

    def build_property_arrays(self): 
        for f in fields(Layer): 
            self.arrays[f.name] = self.get_property_array(f.name) 
