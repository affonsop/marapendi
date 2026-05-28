"""
Module providing classes to model porous layers in electrochemical cells.
"""
from dataclasses import dataclass, field
import numpy as np 
import cantera as ct

from .layer import Layer
from marapendi.models.gas_composition import GasComposition, index_h2, index_o2, index_h2ov, species_indexes, calculate_species_diffusion_coefficient
from marapendi.models.transport_models import PorousGasResistanceModel, DarcyTransportModel
from .water import water_kinematic_viscosity, water_surface_tension, water_molecular_weight, water_dynamic_viscosity, water_density

@dataclass
class PorousLayer(Layer):
    """
    Represents a porous layer in a fuel cell or electrolyzer, defining its
    properties related to gas transport, permeability, capillarity, and thermal behavior.

    Attributes
    ----------
    thickness : float
        Thickness of the porous layer in meters (default is 1e-3 m).
    gas : GasComposition
        Gas composition object representing the gas properties in the layer.
    porosity : float
        Porosity of the layer (0 < porosity < 1).
    effective_gas_diffusion_ratio : float
        Ratio accounting for effective gas diffusion through the porous medium (default is 1).
    pore_diameter : float
        Average pore diameter in meters (default is large, so Knudsen diffusion negligible).
    transport_resistance_model : PorousGasResistanceModel
        Model used to calculate gas transport resistance.
    two_phase_transport_model : DarcyTransportModel
        Model used to compute liquid water flow and capillarity.
    non_wetting_saturation : float
        Fraction of the pore volume occupied by non-wetting phase (0 to 1).
    thermal_conductivity : float
        Thermal conductivity in W/(m·K).
    absolute_permeability : float
        Absolute permeability in m².
    relative_permeability_exponent : float
        Exponent for relative permeability model (default is 3, often used for cubic relationship).
    contact_angle : float
        Contact angle for liquid water in degrees (default is 120°).
    non_wetting_phase : str
        Non-wetting phase (liquid or gas).

    Notes
    -----
    This class internally computes quantities like capillary pressure scaling
    and liquid flow resistance based on current temperature, saturation, and geometry.
    """
    name: str = "porous layer"
    porosity: float = 0.6
    tortuosity: float = 1.
    pore_diameter: float = 20e-6
    absolute_permeability: float = 1e-11 
    relative_permeability_exponent: float = 3
    water_saturation_exponent: float = 3
    contact_angle: float = 120. 
    ionomer_vol_fraction: float = 0.
    ionomer_tortuosity: float = 1.
    breakthrough_pressure: float = 15020
    van_genuchten_m: float = 0.7262
    van_genuchten_n: float = 3.652 
    immobile_non_wetting_saturation: float = 0

    def __post_init__(self): 
        self.sqrt_abs_permeability_porosity = np.sqrt(self.absolute_permeability * self.porosity)
        self.cosinus_contact_angle = np.abs(np.cos(np.pi / 180 * self.contact_angle))

        if self.contact_angle < 90: 
            self.non_wetting_phase = 'gas'
            self.wetting_phase = 'water'
       
    

    
    def calculate_bulk_thermal_resistance(self):
        """
        Computes the thermal resistance of the layer.

        Returns
        -------
        float
            Thermal resistance in m²·K/W.
        """
        return self.thickness / self.bulk_thermal_conductivity 

    

    def set_ionomer_wet_properties(self, ionomer_water_content, temperature):
        pass

    def set_water_film_thickness(self, water_saturation): 
        pass

