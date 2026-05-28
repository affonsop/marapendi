from dataclasses import dataclass, field
from marapendi.tools.tools import Updatable 

@dataclass
class Layer(Updatable): 
    """
    A class representing one layer of an AEM/PEM cell.
    """
    name: str = 'layer'
    thickness: float = 100e-6
    bulk_density: float = 2000.
    bulk_electrical_conductivity: float = 1.e4 
    bulk_specific_heat_capacity: float = 1000. 
    bulk_thermal_conductivity: float = 1.


def get_density(self): 
    return self.bulk_density

def get_tortuosity(self): 
    return self.tortuosity 