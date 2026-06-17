"""
Module providing classes for different membrane permeation models. 
"""
from dataclasses import dataclass
from marapendi.tools import arrhenius_term


@dataclass
class HydrogenPermeationModel:
    """
    A dataclass representing the properties of membrane hydrogen permeability model.
    
    For details, see Trinke et al. (2016)
    
    Attributes:
    -----------
    permeability_reference_temperature : float = 333.15
        The reference temperature for reference permeability coefficients in K. 
    permeability_reference_water_vol_fraction: float = 0.37
        The reference water volume fraction for reference permeability coefficients (n.d.). 
    reference_diffusion_permeability_coefficient: float = 2.95e-17
        Reference hydrogen diffusion permeability coefficient in kmol/(m.s.Pa). 
    diffusion_permeability_activation_energy: float = 27e6
        Diffusion permeability activation energy in J/kmol. 
    reference_convection_permeability_coefficient: float = 9.01e-24
        Reference hydrogen convection permeability coefficient in kmol/(m.s.Pa). 
    convection_permeability_activation_energy: float = 2.7e6
        Convection permeability activation energy in J/kmol.   
    interface_resistance: float = 1.6e12
        Interface resistance in (m.s.Pa)/kmol.
    permeability_correction_factor: float = 1.
        A correction factor to the permeation flux that can be fitted. 

    Methods:
    --------
    calculate_permeability_coefficient(self, temperature, pressure_difference):
        Calculate the effective hydrogen permeability coefficient
    permeation_flux(membrane_thickness, partial_pressure_h2, temperature, 
                    pressure_difference, water_vol_fraction): 
        Calculate the hydrogen permeation flux through the membrane.

    References:
    -----------
    Kang, Z., Pak, M. & Bender, G. Int. J. Hydrogen Energy 46, 15161–15167 (2021).
    Trinke, P. et al. J. Electrochem. Soc. 163, F3164–F3170 (2016).
    """

    # Default values from table II in Trinke et al. (2016)
    permeability_reference_temperature: float = 333.15
    reference_diffusion_permeability_coefficient: float = 2.95e-17
    diffusion_permeability_activation_energy: float = 27e6
    reference_convection_permeability_coefficient: float = 9.01e-24
    convection_permeability_activation_energy: float = 2.7e6
    permeability_reference_water_vol_fraction: float = 0.37
    interface_resistance: float = 1.6e12
    permeability_correction_factor: float = 1.

    def calculate_permeability_coefficient(self, temperature, pressure_difference):
        """
        Calculate the effective hydrogen permeability coefficient. 

        We adopt the model proposed by Trinke et al. (2016) with temperature dependency.
        The effective hydroogne permeability coefficient is calculated as the sum of 
        diffusion and convection contributions. 

        Parameters:
        -----------
        temperature : float
            The temperature in K. 
        
        pressure_difference : float
            The pressure difference between anode and cathode. 
            Positive when the pressure on the hydrogen side is higher. 
         
        Returns:
        --------
        float
            The hydrogen permeability coefficient in kmol/(m·s·Pa).

        References:
        -----------
        Trinke, P. et al. J. Electrochem. Soc. 163, F3164–F3170 (2016).
        """

        diffusion_permeability_coeff = (
            self.reference_diffusion_permeability_coefficient *
            arrhenius_term(
                self.diffusion_permeability_activation_energy,
                temperature,
                self.permeability_reference_temperature
            )
        )

        convection_permeability_coeff = (
            self.reference_convection_permeability_coefficient *
            arrhenius_term(
                self.convection_permeability_activation_energy,
                temperature,
                self.permeability_reference_temperature
            )
        )
        return diffusion_permeability_coeff + convection_permeability_coeff * pressure_difference

    def permeation_flux(self,
                        membrane_thickness: float,
                        partial_pressure_h2: float,
                        temperature: float,
                        pressure_difference: float,
                        water_vol_fraction: float
        ) -> float:
        """
        Calculate the hydrogen permeation flux through the membrane. 

        We adopt the model proposed by Trinke et al. (2016) with temperature dependency, 
        as well as the parameters given in table II of that paper. However, we add an interface 
        resistance in series to avoid overestimating the hydrogen crossover in thin membranes and
        to match (within 10%) experimental data in Kang et al. (2021). A formal validation of this 
        approach would be welcome. 

        Parameters:
        -----------
        membrane_thickness : float 
            The thickness of the membrane in meters (m). 

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

        References:
        -----------
        Kang, Z., Pak, M. & Bender, G. Int. J. Hydrogen Energy 46, 15161–15167 (2021).
        Trinke, P. et al. J. Electrochem. Soc. 163, F3164–F3170 (2016).
        """

        permeability_coefficient = self.calculate_permeability_coefficient(
            temperature,
            pressure_difference
        )

        # The interface resistance is set to reproduce (within 10%) results from Kang et al. (2021).
        # A formal parameter estimation would be better.
        h2_permeability_resistance = (membrane_thickness / permeability_coefficient +
                                        self.interface_resistance)

        return (partial_pressure_h2 / h2_permeability_resistance  *
                    water_vol_fraction  / self.permeability_reference_water_vol_fraction *
                    self.permeability_correction_factor)
