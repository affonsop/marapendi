from dataclasses import dataclass, field
from marapendi.tools.tools import Updatable 

@dataclass(eq=False)
class Layer(Updatable):
    """
    A class representing one layer of an AEM/PEM cell.

    Identity-based equality: layers are unique physical instances in the cell
    model.  Using value equality would cause ambiguous truth-value errors from
    numpy array fields and would incorrectly equate structurally identical but
    physically distinct layers (e.g. anode CL vs cathode CL).
    """
    name: str = 'layer'
    thickness: float = 100e-6
    bulk_density: float = 2000.
    bulk_electrical_conductivity: float = 1.e4
    bulk_specific_heat_capacity: float = 1000.
    bulk_thermal_conductivity: float = 1.

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

