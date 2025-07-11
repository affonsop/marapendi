"""
Module providing a membrane class intended to be the base class for different membrane models. 
"""

import numpy as np
import cantera as ct 
from dataclasses import dataclass, field
from marapendi.tools import calculate_arrhenius_term
from marapendi.water import water_molar_volume
from marapendi.water_balance_models import MembraneWaterBalanceModel
from marapendi.membrane_permeation_models import HydrogenPermeationModel 

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
    hydrogen_permeation_model: HydrogenPermeationModel
        A dataclass representing the properties of membrane hydrogen permeability model.
    water_content: float
        Water content of the membrane. 
    conductivity_correction: float
        Correction factor for the proton conductivity. 
    conductivity_correction: float
        Correction factor for the proton conductivity, to scale the expression from Vetter and Schumacher (2020).  
    conductivity_exp: float
        Exponent for the proton conductivity correlation. 

    Computed Attributes:
    --------------------
    dry_concentration : float
        Concentration of the membrane in mol/m³, computed during initialization.
    dry_molar_volume : float
        Molar volume of the membrane in m³/mol, computed during initialization.

    Methods:
    --------
    water_vol_fraction(water_content, water_molar_volume):
        Calculate the volume fraction of water in the membrane.

    hydrogen_permeation_flux(partial_pressure_h2, hydrogen_permeability):
        Calculate the hydrogen permeation flux through the membrane.
    """

    equivalent_weight: float = 1.1e3
    
    dry_density: float = 1980.
    dry_thickness: float = 25e-6
    h2_permeation_model: HydrogenPermeationModel = field(default_factory=HydrogenPermeationModel)
    water_balance_model: MembraneWaterBalanceModel = field(default_factory=MembraneWaterBalanceModel)
    water_content: float = 14


    def __post_init__(self):
        """
        Compute derived properties of the membrane after initialization.
        """
        self.dry_concentration = self.dry_density / self.equivalent_weight  # kmol/m³
        self.dry_molar_volume = 1. / self.dry_concentration  # m³/kmol
        
    def water_vol_fraction(self, water_content: float, water_molar_volume: float) -> float:
        """
        Calculate the volume fraction of water in the membrane.

        Parameters:
        -----------
        water_content : float
            The water content of the membrane, defined as the number of moles of water 
            per equivalent of the membrane.
        water_molar_volume : float
            The molar volume of water in m³/kmol.

        Returns:
        --------
        float
            The volume fraction of water in the membrane.
        """
        membrane_water_molar_volume = water_molar_volume * water_content
        return membrane_water_molar_volume / (self.dry_molar_volume +
                                               membrane_water_molar_volume)

    def hydrogen_permeation_flux(self,
                                 partial_pressure_h2: float,
                                 temperature: float,
                                 pressure_difference: float,
                                 water_vol_fraction: float,
                                 ) -> float:
        """
        Calculate the hydrogen permeation flux through the membrane. 

        Parameters:
        -----------
        partial_pressure_h2 : float
            The partial pressure of hydrogen in Pascals (Pa).

        temperature : float
            The temperature in K. 
        
        pressure_difference : float
            The pressure difference between anode and cathode. 
            Positive when the pressure on the hydrogen side is higher. 
        
        water_vol_fraction : float
            The membrane water volume fraction.  

        Returns:
        --------
        float
            The hydrogen permeation flux in kmol/(m²·s).
        """

        return self.h2_permeation_model.permeation_flux(self.dry_thickness,
                                                        partial_pressure_h2,
                                                        temperature,
                                                        pressure_difference,
                                                        water_vol_fraction)
    
    def charge_conductivity(self, water_content, temperature, use_water_profile=True, charge='proton'): 
        if charge == 'proton':
            return self.proton_conductivity(water_content, temperature, use_water_profile)
        elif charge == 'hydroxide': 
            return self.hydroxide_conductivity(water_content, temperature)
    
    def charge_resistance(self, water_content, temperature, use_water_profile=True, charge='proton'): 
        return self.dry_thickness / self.charge_conductivity(water_content, temperature, use_water_profile, charge)
    
@dataclass
class PFSA(Membrane):
    conductivity_correction: float = 1
    conductivity_exp: float = 1.5
    conductivity_activation_energy: float = 15e6 
    

    def equilibrium_water_content(self, rh, temperature):
        rh = np.minimum(np.maximum(rh, 0),1)
        return (0.043 + 17.18 * rh - 39.85 * rh**2 + 36 * rh**3)

    def equilibrium_water_content_derivative(self,rh, temperature): 
        rh = np.minimum(np.maximum(rh, 0),1)
        return (17.18 - 79.70 * rh + 108 * rh**2)

    def liquid_equilibrium_water_content(self, temperature): 
        return 9.22 + 0.181 * (temperature - 273.15) # From Goshtasbi et al. (2020)
    
    def proton_conductivity(self, water_content, temperature, use_water_profile=True, water_saturation=0): 
        if use_water_profile:
            fv = self.water_vol_fraction(self.water_balance_model.water_content_profile ,water_molar_volume(temperature))
            return 1/np.mean(1/(self.conductivity_correction * 50 * (np.maximum(fv, 0.11) - 0.1 ) ** self.conductivity_exp * calculate_arrhenius_term(self.conductivity_activation_energy, temperature, 298.15)), axis=0)
        else: 
            fv = self.water_vol_fraction(water_content ,water_molar_volume(temperature))
            return self.conductivity_correction * 50 * (np.maximum(fv, 0.11) - 0.1 ) ** self.conductivity_exp * calculate_arrhenius_term(self.conductivity_activation_energy, temperature, 298.15)

    def proton_resistance(self, water_content, temperature, use_water_profile=True, water_saturation=0):
        liquid_water_content = self.liquid_equilibrium_water_content(temperature)
        return self.dry_thickness / ((1-water_saturation) * self.proton_conductivity(water_content, temperature, use_water_profile)
                                     + water_saturation * self.proton_conductivity(liquid_water_content, temperature, use_water_profile=False))

@dataclass
class FAA3(Membrane):
    dry_density: float = 1310.
    equivalent_weight: float = 1000/1.91
    def hydroxide_conductivity(self, water_content, temperature): 
        return np.interp(temperature, [298.15, 313.15, 333.15, 353.15], [4.10,	5.68,	7.73,	9.17])
    
@dataclass
class PAP85(Membrane):
    dry_density: float = 1220.
    equivalent_weight: float = 1000/2.35

    def hydroxide_conductivity(self, water_content, temperature): 
        return np.interp(temperature, [298.15, 313.15, 333.15, 353.15], [6.63,	8.45,	11.44,	14.87])
    
