"""
Vendored copy of the original (main-branch) marapendi physics model used by
``parameter_estimation.ipynb`` for the JES paper. Kept self-contained and
separate from the new ``marapendi.models``/``marapendi.components`` API so
both can be used side by side.
"""

from .electrochemistry import *
from .water import *
from .fuelcell import *
from .membrane import *
from .gas_composition import *
from .flow_channels import *
from .porous_layers import *
from .transport_models import *
from .electrolyte import *
from .ionomer import *
from .water_balance_models import *
from .membrane_permeation_models import *
from .catalyst_layers import *
from .state import *
from .voltage import *
