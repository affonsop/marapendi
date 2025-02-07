"""
Module providing a membrane class intended to be the base class for different membrane models. 
"""
from dataclasses import dataclass, field
import numpy as np
import cantera as ct 

from coulomb.tools import calculate_arrhenius_term
from coulomb.water import water_saturation_concentration

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
            The hydrogen permeability coefficient in mol/(m·s·Pa).

        References:
        -----------
        Trinke, P. et al. J. Electrochem. Soc. 163, F3164–F3170 (2016).
        """

        diffusion_permeability_coeff = (
            self.reference_diffusion_permeability_coefficient *
            calculate_arrhenius_term(
                self.diffusion_permeability_activation_energy,
                temperature,
                self.permeability_reference_temperature
            )
        )

        convection_permeability_coeff = (
            self.reference_convection_permeability_coefficient *
            calculate_arrhenius_term(
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

@dataclass
class MembraneWaterBalanceModel:
    reference_water_chemical_diffusion_coefficient: float = 4.3e-10
    reference_temperature: float = 303.15
    water_diffusivity_activation_energy: float = 28e6
    # Eq. 31b in Ferrara et al. (2018), equal to 0.2 for Nafion and water contents above 5
    # This assumption tends to overestimate water diffusivity and therefore backdiffusion for 
    # dry membranes. 
    gamma_function_ferrara: float = 0.2 
    absorption_coefficient: float = 3e-4
                                

    def wet_chemical_diffusion_coefficient(self, temperature):
        # Corresponds to D_chem^T in Ferrara et al. (2018)
        return (self.reference_water_chemical_diffusion_coefficient *
                calculate_arrhenius_term(self.water_diffusivity_activation_energy,
                                         temperature,
                                         self.reference_temperature))
    
    def water_diffusivity(self, temperature):
        # According to Ferrara et al. (2018)
        return self.wet_chemical_diffusion_coefficient(temperature) / self.gamma_function_ferrara 
    
    def electroosmotic_drag_speed(self, temperature, current_density, membrane):
        return (0.02 * temperature - 3.86) / 22.5 * current_density / ct.faraday / membrane.dry_concentration
    
    def peclet_number(self, temperature, current_density, membrane):
        return self.electroosmotic_drag_speed(temperature, current_density, membrane) * membrane.dry_thickness / self.water_diffusivity(temperature)
    
    def biot_number(self, absorption_coefficient, temperature, membrane):
        return absorption_coefficient * membrane.dry_thickness / self.water_diffusivity(temperature)

    def water_balance(self, cell):
        k_v = 5e-3
        k_int = self.absorption_coefficient 
        self.k_v_int = k_v * k_int / (k_v + k_int)

        for side in (cell.ca, cell.an):
            side.rh_at_cl_without_crossover = (side.ch.gas.vapor_pressure() / side.cl.gas.saturation_pressure +
                                               (cell.current_density / (2*ct.faraday) / k_v if side == cell.ca else 0))
            side.ch.equiv_water_content = cell.membrane.equilibrium_water_content(
                side.rh_at_cl_without_crossover
            )
            side.membrane_interface_water_content_derivative = cell.membrane.equilibrium_water_content_derivative(
                side.rh_at_cl_without_crossover
            )
            side.membrane_absorption_coefficient_lmbd = (self.k_v_int / 
                                                         cell.membrane.equilibrium_water_content_derivative(side.rh_at_cl_without_crossover))
            
            side.membrane_biot_number =  (1 if side == cell.ca else -1) * self.biot_number(
                side.membrane_absorption_coefficient_lmbd,
                cell.membrane.temperature,
                cell.membrane
            )
        u_d = self.electroosmotic_drag_speed(cell.membrane.temperature, cell.current_density, cell.membrane)
        nondim_vapor_transport_resistance = u_d / k_v / water_saturation_concentration(cell.membrane.temperature)
        cell.membrane.peclet = self.peclet_number(cell.membrane.temperature, cell.current_density, cell.membrane)
        xi = np.linspace(0,np.ones_like(cell.current_density),10)
        
        self.water_content_profile = membrane_water_profile(xi,
                               cell.ca.ch.equiv_water_content,
                               cell.an.ch.equiv_water_content,
                               nondim_vapor_transport_resistance,
                               cell.an.membrane_biot_number,
                               cell.ca.membrane_biot_number,
                               cell.membrane.peclet)
        cell.membrane.water_content = np.mean(self.water_content_profile, axis=0)
        return self.water_content_profile
    
    def cathode_absorption_flux(self, cell): 
        lmbd_ca = self.water_content_profile[-1,:]
        return (cell.ca.membrane_absorption_coefficient_lmbd * (cell.ca.ch.equiv_water_content - lmbd_ca) +
                -self.electroosmotic_drag_speed(cell.membrane.temperature, cell.current_density, cell.membrane) * lmbd_ca * self.k_v_int / self.absorption_coefficient) * cell.membrane.dry_concentration

@dataclass
class SimpleMembraneWaterBalanceModel(MembraneWaterBalanceModel): 

    def water_balance(self, cell): 
        for side in (cell.ca, cell.an):
            side.rh_at_cl_without_crossover = (side.ch.get_vapor_pressure() / side.cl.gas.saturation_pressure +
                                               (cell.current_density / (2*ct.faraday) / side.h2ov_resistance / side.cl.get_saturation_concentration() if side == cell.ca else 0))
            side.membrane_surface_water_content = cell.membrane.equilibrium_water_content(side.rh_at_cl_without_crossover)
        cell.membrane.water_content = 0.5 * sum(side.membrane_surface_water_content for side in (cell.ca, cell.an))


@dataclass
class PemfminesMembraneWaterBalanceModel(MembraneWaterBalanceModel): 

    def water_balance(self, cell): 
        for side in (cell.ca, cell.an):
            side.rh_at_cl_without_crossover = (side.ch.get_vapor_pressure() / side.cl.gas.saturation_pressure +
                                               (cell.current_density / (2*ct.faraday) / side.h2ov_resistance / side.cl.get_saturation_concentration() if side == cell.ca else 0))
            side.membrane_surface_water_content = cell.membrane.equilibrium_water_content(side.rh_at_cl_without_crossover)
        cell.membrane.water_content = 0.5 * sum(side.membrane_surface_water_content for side in (cell.ca, cell.an))


def membrane_water_profile(xi, equiv_lmbd_ca_ch, equiv_lmbd_an_ch, k_v, biot_an, biot_ca, peclet):
    """
    Calculate membrane water profile for constant water absorption and diffusivity coefficients.
    Based on the work of Ferrara et al. (2018), with an extension to consider interface resitances 
    (Biot numbers for cathode and anode) and vapor transport resistance between the channel and the
    catalyst layer.     

    Parameters:
    -----------
    xi : float
        Non-dimensinal membrane thickness, between 0 (anode) and 1 (cathode). 
    equiv_lmbd_ca_ch : float
        Equivalent membrane water content corresponding to the RH at
        the cathode channel calculated at the catalyst layer temperature.
    equiv_lmbd_ca_ch : float
        Equivalent membrane water content corresponding to the RH at
        the anode channel calculated at the catalyst layer temperature.
    k_v : float
        Non-dimensional vapor transport resistance. 
    biot_ca : float
        Biot number relating water absorption/desorption at the cathode 
        side and diffusion in the bulk membrane. 
    biot_an : float
        Biot number relating water absorption/desorption at the cathode 
        side and diffusion in the bulk membrane.
    peclet : float
        Peclet number relating electroosmotic drag to water diffusion.
    """
    exp_peclet = np.exp(peclet)

    A = np.exp(peclet * xi) + peclet/biot_an - 1
    B = peclet/biot_ca*exp_peclet + peclet/biot_an + (1+k_v)*(exp_peclet-1)
    C = (1-k_v)*peclet/biot_ca*exp_peclet + (1+k_v)*peclet/biot_an + (1-k_v**2)*(exp_peclet-1)

    return (A * (equiv_lmbd_ca_ch - equiv_lmbd_an_ch) + equiv_lmbd_an_ch * B)/C
    

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
    density: float = 1980.
    dry_thickness: float = 25e-6
    h2_permeation_model: HydrogenPermeationModel = field(default_factory=HydrogenPermeationModel)
    water_balance_model: MembraneWaterBalanceModel = field(default_factory=MembraneWaterBalanceModel)
    water_content: float = 14

    def __post_init__(self):
        """
        Compute derived properties of the membrane after initialization.
        """
        self.dry_concentration = self.density / self.equivalent_weight  # mol/m³
        self.dry_molar_volume = 1. / self.dry_concentration  # m³/mol

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
    
    def equilibrium_water_content(self, rh):
        return (0.043 + 17.18 * rh - 39.85 * rh**2 + 36 * rh**3)

    def equilibrium_water_content_derivative(self,rh): 
        return (17.18 - 79.70 * rh + 108 * rh**2)

    def proton_conductivity(self, temperature, water_vol_fraction, water_content): 
        return (0.539 * np.maximum(water_content, 1) - 0.326) * calculate_arrhenius_term(10e6, temperature, 303.15)

    def proton_resistance(self, temperature, water_vol_fraction, water_content): 
        return self.dry_thickness / self.proton_conductivity(temperature, water_vol_fraction, water_content)