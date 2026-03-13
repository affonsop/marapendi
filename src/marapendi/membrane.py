"""
Module providing a membrane class intended to be the base class for different membrane models. 
"""

import numpy as np
import cantera as ct 
from dataclasses import dataclass, field
from marapendi.tools import arrhenius_term
from marapendi.water import water_molar_volume
from marapendi.water_balance_models import MembraneWaterBalanceModel
from marapendi.membrane_permeation_models import HydrogenPermeationModel 

@dataclass
class Membrane:
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

    equivalent_weight: float = 1.1e3
    
    dry_density: float = 1980.
    dry_thickness: float = 25e-6
    h2_permeation_model: HydrogenPermeationModel = field(default_factory=HydrogenPermeationModel)
    water_balance_model: MembraneWaterBalanceModel = field(default_factory=MembraneWaterBalanceModel)
    water_content: float = 14
    relaxation_time_constant: float = 0.067 # Default from Grimaldi et al. (2023)
    relaxation_time_activation_energy: float = 28e6 # Default from Grimaldi et al. (2023)
    uptake_relaxed_fraction_constant: float = 0.014 # Default from Grimaldi et al. (2023)
    def __post_init__(self):
        """
        Compute derived properties of the membrane after initialization.
        """
        self.dry_concentration = self.dry_density / self.equivalent_weight  # kmol/m³
        self.dry_molar_volume = 1. / self.dry_concentration  # m³/kmol
        self.surface_concentration = self.dry_concentration * self.dry_thickness

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
        """
        Calculate the charge conductivity of the membrane.

        Parameters
        ----------
        water_content : float
            The water content of the membrane. Used if use_water_profile is False.
        temperature : float
            The temperature in Kelvin (K).
        use_water_profile : bool, optional
            Whether to use the water profile in calculations (default is True).
        charge : str, optional
            The type of charge, either 'proton' or 'hydroxide' (default is 'proton').

        Returns
        -------
        float
            The charge conductivity of the membrane in Siemens per meter (S/m).
        """
        if charge == 'proton':
            return self.proton_conductivity(water_content, temperature, use_water_profile)
        elif charge == 'hydroxide': 
            return self.hydroxide_conductivity(water_content, temperature)
    
    def charge_resistance(self, water_content, temperature, use_water_profile=True, charge='proton'): 
        """
        Calculate the charge resistance of the membrane.

        Parameters
        ----------
        water_content : float
            The water content of the membrane. Used if use_water_profile is False.
        temperature : float
            The temperature in Kelvin (K).
        use_water_profile : bool, optional
            Whether to use the water profile in calculations (default is True).
        charge : str, optional
            The type of charge, either 'proton' or 'hydroxide' (default is 'proton').

        Returns
        -------
        float
            The charge resistance of the membrane in ohm square meters (Ohm.m²).
        """
        return self.dry_thickness / self.charge_conductivity(water_content, temperature, use_water_profile, charge)
    
@dataclass
class PFSA(Membrane):
    """
    A class representing a Perfluorosulfonic Acid (PFSA) membrane, extending the Membrane class.
    This class includes properties and methods for calculating water content and proton conductivity.

    Attributes
    ----------
    conductivity_correction : float
        Correction factor for the proton conductivity, to scale the expression from Vetter and Schumacher (2020). Default is 1.
    conductivity_exp : float
        Exponent for the proton conductivity correlation. Default is 1.5.
    conductivity_activation_energy : float
        Activation energy for proton conductivity in Joules. Default is 15e6.
    phi : float
        Contribution of relaxation phenomena to the ionomer water uptake, according to Goshtasbi et al. (2019).

    Methods
    -------
    equilibrium_water_content(rh, temperature)
        Calculate the equilibrium water content based on relative humidity and temperature.
    equilibrium_water_content_derivative(rh, temperature)
        Calculate the derivative of the equilibrium water content with respect to relative humidity.
    liquid_equilibrium_water_content(temperature)
        Calculate the liquid equilibrium water content based on temperature.
    proton_conductivity(water_content, temperature, use_water_profile=True, water_saturation=0)
        Calculate the proton conductivity based on water content and temperature.
    proton_resistance(water_content, temperature, use_water_profile=True, water_saturation=0)
        Calculate the proton resistance through the membrane.
    
    References
    ----------
    Springer, T. E. et al. J. Electrochem. Soc. 138, 2334 (1991).
    Grimaldi et al. J. Power Sources (2023).
    Goshtasbi et al. J. Electrochem. Soc. 2019, 166 (7), F3154.
    """

    conductivity_correction: float = 1
    conductivity_exp: float = 1.5
    conductivity_activation_energy: float = 15e6 
    phi: float = 0.15

    def equilibrium_water_content(self, rh, temperature, s_relax=None):
            """
            Calculate the equilibrium water content based on relative humidity and temperature.
            Uses the polynomial interpolation obtained by Springer et al. (1991) for Nafion N117 at 30ºC. 

            Parameters
            ----------
            rh : float
                Relative humidity, a value between 0 and 1.
            temperature : float
                Temperature in Kelvin (K).
            s_relax : float
                Membrane relaxation term, a value between 0 and 1.

            lmbd : float
                Membrane water content 

            Returns
            -------
            float
                The equilibrium water content of the membrane.
            
            References
            ----------
            Springer, T. E. et al. J. Electrochem. Soc. 138, 2334 (1991).
            Grimaldi et al. J. Power Sources (2023).
            Goshtasbi et al. J. Electrochem. Soc. 2019, 166 (7), F3154.
            """
            rh = np.minimum(np.maximum(rh, 0), 1)
            lmbd_eq_relaxed = (0.043 + 17.18 * rh - 39.85 * rh**2 + 36 * rh**3)
            
            return ((1 - self.phi) * lmbd_eq_relaxed + s_relax) if s_relax is not None else lmbd_eq_relaxed 

    def equilibrium_water_content_derivative(self, rh, temperature, s_relax=None):
        """
        Calculate the derivative of the equilibrium water content with respect to relative humidity.
        Uses the polynomial interpolation obtained by Springer et al. (1991) for Nafion N117 at 30ºC. 

        Parameters
        ----------
        rh : float
            Relative humidity, a value between 0 and 1.
        temperature : float
            Temperature in Kelvin (K).
        s_relax : float
            Membrane relaxation term, a value between 0 and 1.
        lmbd : float
            Membrane water content 

        Returns
        -------
        float
            The derivative of the equilibrium water content with respect to relative humidity.

        References
        ----------
        Springer, T. E. et al. J. Electrochem. Soc. 138, 2334 (1991).
        Grimaldi et al. J. Power Sources (2023).
        Goshtasbi et al. J. Electrochem. Soc. 2019, 166 (7), F3154.
        """
        rh = np.minimum(np.maximum(rh, 0), 1)
        d_lmbd_eq_relaxed = (17.18 - 79.70 * rh + 108 * rh**2)
        return ((1 - self.phi) * d_lmbd_eq_relaxed + s_relax) if s_relax is not None else d_lmbd_eq_relaxed 
    
    


    def liquid_equilibrium_water_content(self, temperature):
        """
        Calculate the liquid equilibrium water content based on temperature.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin (K).

        Returns
        -------
        float
            The liquid equilibrium water content of the membrane.

        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        return 9.22 + 0.181 * (temperature - 273.15) # From Goshtasbi et al. (2020)

    def proton_conductivity(self, water_content, temperature, use_water_profile=True, water_saturation=0):
        """
        Calculate the proton conductivity based on water content, water saturation and temperature.

        Parameters
        ----------
        water_content : float
            The water content of the membrane. Used if use_water_profile is set to False. 
        temperature : float
            Temperature in Kelvin (K).
        use_water_profile : bool, optional
            Whether to use the membrane water profile in calculations (default is True).
        water_saturation : float, optional
            The water saturation level (default is 0).

        Returns
        -------
        float
            The proton conductivity of the membrane in Siemens per meter (S/m).
        """
        if use_water_profile:
            fv = self.water_vol_fraction(self.water_balance_model.water_content_profile, water_molar_volume(temperature))
            return 1/np.mean(1/(self.conductivity_correction * 50 * (np.maximum(fv, 0.11) - 0.1 ) ** self.conductivity_exp * arrhenius_term(self.conductivity_activation_energy, temperature, 298.15)), axis=0)
        else:
            fv = self.water_vol_fraction(water_content, water_molar_volume(temperature))
            return self.conductivity_correction * 50 * (np.maximum(fv, 0.11) - 0.1 ) ** self.conductivity_exp * arrhenius_term(self.conductivity_activation_energy, temperature, 298.15)

    def proton_resistance(self, water_content, temperature, use_water_profile=True, water_saturation=0):
        """
        Calculate the proton resistance through the membrane. 
        Considers an average conductivity from liquid and vapor equilibrated conditions if liquid saturation is greater 
        than zero.

        Parameters
        ----------
        water_content : float
            The water content of the membrane.
        temperature : float
            Temperature in Kelvin (K).
        use_water_profile : bool, optional
            Whether to use the water profile in calculations (default is True).
        water_saturation : float, optional
            The water saturation level (default is 0).

        Returns
        -------
        float
            The proton resistance of the membrane in ohm square meters (Ω·m²).
        """
        liquid_water_content = self.liquid_equilibrium_water_content(temperature)
        liquid_equilibrated_conductivity =  self.proton_conductivity(liquid_water_content, temperature, use_water_profile=False)
        vapor_equilibrated_conductivity = self.proton_conductivity(water_content, temperature, use_water_profile)
        average_conductivity = (1-water_saturation) * vapor_equilibrated_conductivity + water_saturation * liquid_equilibrated_conductivity
        return self.dry_thickness / average_conductivity

@dataclass
class FAA3(Membrane):
    """
    A class representing an FAA3 membrane, extending the Membrane class.
    This class includes properties and methods for calculating hydroxide conductivity.

    References
    ----------
    Eon Chae, J. et al. J. Ind. Eng. Chem. 133, 255–262 (2024)
    Luo, X. et al. J. Memb. Sci. 598, 117680 (2020)
    Khalid, H. et al. Membranes (Basel) 12, 989 (2022).

    Attributes
    ----------
    dry_density : float
        Density of the membrane in kg/m³. Default is 1310 kg/m³.
    equivalent_weight : float
        Equivalent weight of the membrane in kg/kmol. Default is 1000/1.91 kg/kmol.

    Methods
    -------
    hydroxide_conductivity(water_content, temperature)
        Calculate the hydroxide conductivity based on water content and temperature.
    """
    # Data from Luo et al. (2020), table 1. 
    dry_density: float = 1310.
    equivalent_weight: float = 1000/1.91

    def hydroxide_conductivity(self, water_content, temperature):
        """
        Calculate the hydroxide conductivity of the membrane based on water content and temperature.

        Parameters
        ----------
        water_content : float
            The water content of the membrane (not used in calculation but kept for consistency).
        temperature : float
            The temperature in Kelvin (K).

        Returns
        -------
        float
            The hydroxide conductivity of the membrane in Siemens per meter (S/m).
        """
        # Room-temperature conductivity from Luo et al. (2020) with
        # activation energy from Khalid et al. (2022) for FAA3-50. Liquid-equilibrated.
        return 3.1 * arrhenius_term(activation_energy=11.1e6,
                                              temperature=temperature,
                                              reference_temperature=298.15)

@dataclass
class PAP85(Membrane):
    """
    A class representing a PAP85 membrane, extending the Membrane class.
    This class includes properties and methods for calculating hydroxide conductivity.

    References
    ----------
    Eon Chae, J. et al. J. Ind. Eng. Chem. 133, 255–262 (2024)
    Luo, X. et al. J. Memb. Sci. 598, 117680 (2020)
    Khalid, H. et al. Membranes (Basel) 12, 989 (2022).

    Attributes
    ----------
    dry_density : float
        Density of the membrane in kg/m³. Default is 1220 kg/m³.
    equivalent_weight : float
        Equivalent weight of the membrane in kg/kmol. Default is 1000/2.35 kg/kmol.

    Methods
    -------
    hydroxide_conductivity(water_content, temperature)
        Calculate the hydroxide conductivity based on water content and temperature.
    """
    # Data from Luo et al. (2020), table 1. 
    dry_density: float = 1220.
    equivalent_weight: float = 1000/2.35

    def hydroxide_conductivity(self, water_content, temperature):
        """
        Calculate the hydroxide conductivity of the membrane based on water content and temperature.

        Parameters
        ----------
        water_content : float
            The water content of the membrane (not used in calculation but kept for consistency).
        temperature : float
            The temperature in Kelvin (K).

        Returns
        -------
        float
            The hydroxide conductivity of the membrane in Siemens per meter (S/m).
        """
        # Room-temperature conductivity for liquid-equilibrated from Luo et al. (2020) with
        # activation energy from Khalid et al. (2022) for PAP-20. Liquid-equilibrated.
        return 5.8 * arrhenius_term(activation_energy=22.5e6,
                                              temperature=temperature,
                                              reference_temperature=298.15)
    
@dataclass
class SustainionX3750RT(Membrane):
    """
    A class representing a Sustainion X37-50 RT membrane membrane, extending the Membrane class.
    This class includes properties and methods for calculating hydroxide conductivity.

   
    Attributes
    ----------
    dry_density : float
        Density of the membrane in kg/m³. Default is 1220 kg/m³.
    equivalent_weight : float
        Equivalent weight of the membrane in kg/kmol. Default is 1000/2.35 kg/kmol.

    Methods
    -------
    hydroxide_conductivity(water_content, temperature)
        Calculate the hydroxide conductivity based on water content and temperature.
    """
    # Data from Luo et al. (2020), table 1. 
    dry_density: float = 1220.
    equivalent_weight: float = 1000/2.35
    dry_thickness: float = 50e-6 
    ref_hydroxide_conductivity: float = 11.6
    conductivity_activation_energy: float = 10.7e6
    def hydroxide_conductivity(self, water_content, temperature):
        """
        Calculate the hydroxide conductivity of the membrane based on water content and temperature.

        Parameters
        ----------
        water_content : float
            The water content of the membrane (not used in calculation but kept for consistency).
        temperature : float
            The temperature in Kelvin (K).

        Returns
        -------
        float
            The hydroxide conductivity of the membrane in Siemens per meter (S/m).
        """
        # Room-temperature conductivity for liquid-equilibrated from Luo et al. (2020) with
        # activation energy from Khalid et al. (2022) for PAP-20. Liquid-equilibrated.
        return self.ref_hydroxide_conductivity * arrhenius_term(activation_energy=self.conductivity_activation_energy,
                                              temperature=temperature,
                                              reference_temperature=333.15)
    