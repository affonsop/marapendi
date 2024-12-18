"""
Module providing a membrane class intended to be the base class for different membrane models. 
"""
from dataclasses import dataclass
import numpy as np 

from coulomb.tools import calculate_arrhenius_term
@dataclass
class Membrane:
    """
    A dataclass representing the properties of a proton exchange membrane (PEM) 
    and methods for calculating water volume fraction, hydrogen permeability, 
    and hydrogen permeation flux.

    Attributes:
    -----------
    equivalent_weight : float
        Equivalent weight of the membrane in kg/kmol. Default is 1100 kg/kmol.
    density : float
        Density of the membrane in kg/m³. Default is 1980 kg/m³.
    thickness : float
        Thickness of the membrane in meters (m). Default is 25 µm.

    Computed Attributes:
    --------------------
    membrane_concentration : float
        Concentration of the membrane in mol/m³, computed during initialization.
    membrane_molar_volume : float
        Molar volume of the membrane in m³/mol, computed during initialization.

    Methods:
    --------
    water_vol_fraction(water_content, water_molar_volume):
        Calculate the volume fraction of water in the membrane.
        
    hydrogen_permeation_flux(partial_pressure_h2, hydrogen_permeability):
        Calculate the hydrogen permeation flux through the membrane.

    References:
    -----------
    Kang, Z., Pak, M. & Bender, G. Int. J. Hydrogen Energy 46, 15161–15167 (2021).
    Trinke, P. et al. J. Electrochem. Soc. 163, F3164–F3170 (2016).
  

    """

    equivalent_weight: float = 1.1e3
    density: float = 1980.
    thickness: float = 25e-6

    def __post_init__(self):
        """
        Compute derived properties of the membrane after initialization.
        """
        self.membrane_concentration = self.density / self.equivalent_weight  # mol/m³
        self.membrane_molar_volume = 1. / self.membrane_concentration  # m³/mol

    def water_vol_fraction(self, water_content: float, water_molar_volume: float) -> float:
        """
        Calculate the volume fraction of water in the membrane.

        Parameters:
        -----------
        water_content : float
            The water content of the membrane, defined as the number of moles of water 
            per equivalent of the membrane.
        water_molar_volume : float
            The molar volume of water in m³/mol.

        Returns:
        --------
        float
            The volume fraction of water in the membrane.
        """
        membrane_water_molar_volume = water_molar_volume * water_content
        return membrane_water_molar_volume / (self.membrane_molar_volume + membrane_water_molar_volume)
    
    def hydrogen_permeation_flux(self, 
                                 partial_pressure_h2: float, 
                                 temperature: float, 
                                 pressure_difference: float, 
                                 water_vol_fraction: float, 
                                 correction_factor: float = 1) -> float:
        """
        Calculate the hydrogen permeation flux through the membrane. 

        We adopt the model proposed by Trinke et al. (2016) with temperature dependancy, 
        as well as the parameters given in table II of that paper. However, we add an interface 
        resistance in series to avoid overestimating the hydrogen crossover in thin membranes and
        to match (within 10%) experimental data in Kang et al. (2021). A formal validation of this 
        approach would be welcome. 

        Parameters:
        -----------
        partial_pressure_h2 : float
            The partial pressure of hydrogen in Pascals (Pa).
        
        hydrogen_permeability : float
            The hydrogen permeability in mol·m/(s·Pa).

        temperature : float
            The temperature in K. 
        
        pressure_difference : float
            The pressure difference between anode and cathode. 
            Positive when the pressure on the hydrogen side is higher. 
        
        water_vol_fraction : float
            The membrane water volume fraction. 

        correction_factor : float
            A fitted correction factor to account for deviation from reference parameters. 

        Returns:
        --------
        float
            The hydrogen permeation flux in mol/(m²·s).

        References:
        -----------
        Kang, Z., Pak, M. & Bender, G. Int. J. Hydrogen Energy 46, 15161–15167 (2021).
        Trinke, P. et al. J. Electrochem. Soc. 163, F3164–F3170 (2016).
        """
        # Values from table II in Trinke et al. (2016)
        reference_temperature= 333.15
        reference_diffusion_permeability_coefficient = 2.95e-14
        diffusion_permeability_activation_energy = 27e6
        reference_convection_permeability_coefficient = 9.01e-21
        convection_permeability_activation_energy = 2.7e6
        reference_water_vol_fraction = 0.37


        diffusion_permeability_coeff = (reference_diffusion_permeability_coefficient * 
                                        calculate_arrhenius_term(diffusion_permeability_activation_energy, 
                                                                 temperature, 
                                                                 reference_temperature)) 

        convection_permeability_coeff = (reference_convection_permeability_coefficient * 
                                        calculate_arrhenius_term(convection_permeability_activation_energy, 
                                                                 temperature, 
                                                                 reference_temperature))   
        
        h2_permeability_coefficient = (diffusion_permeability_coeff + convection_permeability_coeff * pressure_difference) 

        # The interface resistance is set to reproduce (within 10%) results from Kang et al. (2021). 
        # A formal parameter estimation would be better. 
        h2_interface_resistance = 1.6e9  
        h2_permeability_resistance = (self.thickness / h2_permeability_coefficient + h2_interface_resistance)
        return (partial_pressure_h2 / h2_permeability_resistance  *
                 water_vol_fraction  / reference_water_vol_fraction * 
                 correction_factor)
