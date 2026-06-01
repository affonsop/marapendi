"""
Module providing a membrane class intended to be the base class for different membrane models. 
"""

import numpy as np
import cantera as ct 
from dataclasses import dataclass, field, fields as dataclass_fields
from marapendi.components.ionomer import Ionomer, PFSAIonomer
from marapendi.components.porous_layers import PorousLayer

@dataclass
class Membrane(PorousLayer, Ionomer):
    """
    A base dataclass representing the properties of a proton/anion exchange membrane 
    and methods for calculating water volume fraction, hydrogen permeability, 
    and hydrogen permeation flux.

    Attributes:
    -----------
    
    equivalent_weight : float
        Equivalent weight of the membrane in kg/kmol. Default is 1100 kg/kmol.
    dry_density : float
        Density of the membrane in kg/m³. Default is 1980 kg/m³.
    dry_thickness : float
        Thickness of the membrane in meters (m). Default is 25 µm.
    h2_permeation_model: HydrogenPermeationModel
        A dataclass representing the properties of membrane hydrogen permeability model.
    water_content: float
        Water content of the membrane. 
    water_balance_model: MembraneWaterBalanceModel
        Water balance model allowing to calculate water contents in the membrane and CL. 
    reference_water_diffusivity : float, optional
        Reference value for water diffusivity, in m2/s (default is 4.3e-10).
    reference_absorption_coefficient : float, optional
        Reference value for the absorption coefficient (default is 1e-5).
    reference_temperature : float, optional
        Reference temperature for calculations, in Kelvin (default is 353.15 K).
    water_diffusivity_activation_energy : float, optional
        Activation energy for water diffusivity, in J/kmol (default is 20e6).
    water_absorption_activation_energy : float, optional
        Activation energy for water absorption, in J/kmol (default is 20e6).

    Computed Attributes:
    --------------------
    dry_concentration : float
        Concentration of the membrane in kmol/m³, computed during initialization.
    dry_molar_volume : float
        Molar volume of the membrane in m³/kmol, computed during initialization.

    Methods:
    --------
    water_vol_fraction(water_content, water_molar_volume):
        Calculate the volume fraction of water in the membrane.

    hydrogen_permeation_flux(partial_pressure_h2, hydrogen_permeability):
        Calculate the hydrogen permeation flux through the membrane.
    """
    
    thickness: float
    eps_ion: float = 1.
    tort_ion: float = 1.
    ionomer: Ionomer = field(default=None)
    thermal_conductivity: float = 0.9
    specific_heat_capacity: float = 2000.
    
    def __eq__(self, other):
        """Use identity comparison — layers are unique instances in the cell model.
        This avoids ambiguous truth-value errors from numpy array fields during
        membership tests like `layer in self.ionomer_layers`."""
        return self is other

    def __hash__(self):
        return id(self)


    def __post_init__(self):
        """Copy all ionomer fields onto self so ionomer properties are
        directly accessible (e.g. self.dry_density, self.equivalent_weight).
        Skipped if no ionomer was supplied."""
        if self.ionomer is not None:
            for f in dataclass_fields(self.ionomer):
                setattr(self, f.name, getattr(self.ionomer, f.name))
        PorousLayer.__post_init__(self)
        Ionomer.__post_init__(self)

@dataclass
class PFSAMembrane(Membrane, PFSAIonomer):
    pass