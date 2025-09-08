import numpy as np
import cantera as ct 
from dataclasses import dataclass, field
from marapendi.tools import calculate_arrhenius_term

@dataclass
class MembraneWaterBalanceModel:
    """
    A class representing a membrane water balance model with various parameters related to water diffusivity and absorption.

    Attributes
    ----------
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
    sorption_activity_driving_force : bool, optional
        Boolean flag indicating whether water activity is the driving force for water absorption (default is False).
        If false, water content difference is considered as the driving force. 
    eod_parallel_to_sorption : bool, optional
        Boolean flag indicating if electro-osmotic drag is parallel to water absorption/desorption (default is False).
        If True, electro-osmotic drag is added to the water absorption flux on the RHS of the water balance boundary conditions.  

    Notes
    -----
    The class is based on the equations and assumptions in Ferrara et al. (2018), while accounting 
    for gas transport resistance and non-equilibrium conditions at the membrane interface.

    References:
    -----------
    Ferrara, A. et al. J. Power Sources 390, 197–207 (2018).
    """
    reference_water_diffusivity: float = 4.3e-10
    reference_absorption_coefficient: float = 1e-5
    reference_temperature: float = 353.15
    water_diffusivity_activation_energy: float = 20e6
    water_absorption_activation_energy: float = 20e6
    sorption_activity_driving_force: bool = False
    eod_parallel_to_sorption: bool = False                            

    def calculate_water_absorption_coefficient(self, temperature): 
        """
        Calculate the water absorption coefficient based on temperature.

        Parameters
        ----------
        temperature : float 
            The temperature at which to calculate the water absorption coefficient (K).

        Returns
        -------
        float
            The calculated water absorption coefficient (m/s).

        Notes
        -----
        The calculation uses the reference water absorption coefficient and the Arrhenius term.
        """
        return  (
                    self.reference_absorption_coefficient * 
                    calculate_arrhenius_term(
                        self.water_absorption_activation_energy, 
                        temperature, 
                        self.reference_temperature
                    )
                )
    
    def calculate_water_diffusivity(self, temperature):
        """
        Calculate the water diffusivity based on temperature.

        Parameters
        ----------
        temperature : float
            The temperature at which to calculate the water diffusivity (K).

        Returns
        -------
        float
            The calculated water diffusivity (m2/s).

        Notes
        -----
        The calculation uses the reference water diffusivity and the Arrhenius term.
        """
        return (self.reference_water_diffusivity *
                calculate_arrhenius_term(self.water_diffusivity_activation_energy,
                                        temperature,
                                        self.reference_temperature))
    
    def calculate_electroosmotic_drag_coefficient(self, temperature, water_content):
        """
        Calculate the electroosmotic drag coefficient based on temperature and water content.

        Parameters
        ----------
        temperature : float 
            The temperature at which to calculate the EOD coefficient (K).
        water_content : float
            The membrane water content (n.d.).

        Returns
        -------
        float
            The calculated electroosmotic drag coefficient (n.d.).
        """
        return (0.02 * temperature - 3.86) / 22.5 * water_content
    
    def calculate_electroosmotic_drag_speed(self, temperature, current_density, membrane):
        """
        Calculate the electroosmotic drag speed using temperature, current density, and membrane properties.

        Parameters
        ----------
        temperature : float
            The temperature at which to calculate the drag speed (K).
        current_density : float
            The current density crossing the membrane (A/m2).
        membrane : object
            An object with membrane properties, including dry concentration.

        Returns
        -------
        float
            The calculated electroosmotic drag speed (m/s).
        """
        # Calculate and return the electroosmotic drag speed
        return (self.calculate_electroosmotic_drag_coefficient(temperature, 1) *
                current_density / ct.faraday /
                membrane.dry_concentration)
    
    def calculate_peclet_number(self, temperature, current_density, membrane, water_diffusivity):
        """
        Calculate the Peclet number using electroosmotic drag speed, membrane thickness, and water diffusivity.

        Parameters
        ----------
        temperature : float
            The temperature at which to calculate the Peclet number, in Kelvin (K).
        current_density : float
            The current density crossing the membrane (A/m2).
        membrane : object
            An object with membrane properties, including dry thickness (m).
        water_diffusivity : float
            The adsorbed water diffusivity (m²/s).

        Returns
        -------
        float
            The calculated Peclet number, dimensionless.
        """
        # Calculate and return the Peclet number
        return (self.calculate_electroosmotic_drag_speed(temperature, current_density, membrane) *
                membrane.dry_thickness / water_diffusivity)
    
    def calculate_biot_number(self, absorption_coefficient, membrane, water_diffusivity):
        """
        Calculate the Biot number using absorption coefficient, membrane thickness, and water diffusivity.

        Parameters
        ----------
        absorption_coefficient : float
            The water absorption coefficient (m/s).
        membrane : object
            An object with membrane properties, including dry thickness.
        water_diffusivity : float
            The membrane water diffusivity (m²/s).

        Returns
        -------
        float
            The calculated Biot number, dimensionless.
        """
        return absorption_coefficient * membrane.dry_thickness / water_diffusivity

    def estimate_equilibrium_water_contents(self, cell):
        """
        Estimates the equilibrium water content and its derivative for both the cathode and anode sides of the cell.

        Parameters
        ----------
        cell : FuelCell
            An object representing the cell with properties ca (cathode) and an (anode), each being a CellSide object.

        Notes
        -----
        The function updates the cell sides (ca and an) with estimated water content and its derivative.
        The calculations assume no water crossover from cathode to anode for RH at catalyst layer.
        """

        for side in (cell.ca, cell.an):
            # Calculate RH at catalyst layer supposing no water crossover from cathode to anode  
            side.rh_at_cl_without_crossover = np.minimum(
                (side.ch.vapor_concentration() + ((cell.current_density / (2*ct.faraday) * side.h2ov_transport_resistance)
                                                if side == cell.ca else np.zeros_like(cell.current_density)))
                
                / side.cl.saturation_concentration(),
                1
            )

            # Get estimated values for equilibrium water content and its derivative in CL 
            side.est_water_content = cell.membrane.equilibrium_water_content(
                side.rh_at_cl_without_crossover, cell.membrane.temperature, 
            )
            side.est_water_content_derivative = cell.membrane.equilibrium_water_content_derivative(
                side.rh_at_cl_without_crossover, cell.membrane.temperature, 
            )

    def update_water_contents(self, cell):
        """
        Set the water content values for different parts of the cell, including catalyst layers and membrane.

        Parameters
        ----------
        cell : FuelCell object
            An object representing the cell with properties ca (cathode), an (anode), membrane, etc.,
            each having attributes related to water content and transport properties.

        Notes
        -----
        This function updates the following attributes of the cell:
        - Equilibrium water content in the cathode and anode catalyst layers.
        - Average water content in the membrane.
        - Ionomer water content in the cathode and anode catalyst layers.
        """

        # Set the average water content in the membrane
        cell.membrane.water_content = np.mean(self.water_content_profile, axis=0)

        # Set water content at the interface of each catalyst layer 
        cell.ca.cl.memb_interface_water_content = self.water_content_profile[-1,:]
        cell.an.cl.memb_interface_water_content = self.water_content_profile[0,:]

        # Calculate equilibrium water contents at the CL
        for side in (cell.ca, cell.an):
            side.cl.eq_water_content = side.cl.memb_interface_water_content - side.membrane_water_flux / self.absorption_coefficient / cell.membrane.dry_concentration
        
    def update_water_profile(self, cell): 
        """
        Calculate the water content profile across the cell based on various parameters.

        Parameters
        ----------
        cell : FuelCell object
            An object representing the cell with properties related to water content, transport,
            membrane properties, and other physical and structural properties.

        Notes
        -----
        This function computes the water content profile based on an extension of the work of Ferrara et al. (2018) 
        accounting for non-equilibrium conditions at the membrane interfaces. 
        Parameters are calculated in the water_balance functions. 
        """
        # Generate evenly spaced points over the interval
        xi = np.linspace(0, np.ones_like(cell.current_density), 20)
        ePe = np.exp(cell.membrane.Pe)
        ePexi = np.exp(xi * cell.membrane.Pe)

        # Calculate the water content profile using a detailed mathematical formula
        self.water_content_profile = (
            (
                cell.an.est_water_content * ((ePe - ePexi) * (1 - cell.ca.alpha * cell.ca.Pe_over_mod_Bi) + ePe * cell.ca.Pe_over_mod_Bi) +
                cell.ca.est_water_content * ((ePexi - 1) * (1 + cell.an.alpha * cell.an.Pe_over_mod_Bi) + cell.an.Pe_over_mod_Bi)
            ) / (
                (ePe - 1) * (
                    1 + cell.an.alpha * cell.an.Pe_over_mod_Bi - cell.ca.alpha * cell.ca.Pe_over_mod_Bi +
                    -cell.ca.alpha * cell.an.alpha * cell.an.Pe_over_mod_Bi * cell.ca.Pe_over_mod_Bi
                ) +
                cell.an.Pe_over_mod_Bi + ePe * cell.ca.Pe_over_mod_Bi +
                cell.an.Pe_over_mod_Bi * cell.ca.Pe_over_mod_Bi * (ePe * cell.an.alpha - cell.ca.alpha)
            )
        )
    
    def update_non_dimensional_parameters(self, cell):
        """
        Calculate various non-dimensional parameters related to water transport and equilibrium in a cell.

        Parameters
        ----------
        cell : FuelCell object
            An object representing the cell with properties related to water content, transport,
            membrane properties, and other physical and structural properties.

        Notes
        -----
        This function computes various non-dimensional parameters used in the analysis of water balance within the cell,
        including the Peclet number, non-dimensional water vapor resistance, Biot number, and others.
        """
        # Calculate Peclet number for the membrane
        cell.membrane.Pe = self.calculate_peclet_number(cell.membrane.temperature, cell.current_density, cell.membrane, self.water_diffusivity)

        for side in (cell.ca, cell.an):
            # Calculate non-dimensional water vapor resistance based on a water content driving force
            side.R_v_star = side.h2ov_transport_resistance / (side.cl.saturation_concentration() * cell.K_mb) * side.est_water_content_derivative

            # Calculate Biot number
            side.Bi = self.calculate_biot_number(
                self.absorption_coefficient / (side.est_water_content_derivative if self.sorption_activity_driving_force else 1),
                cell.membrane, self.water_diffusivity
            )

            # Calculate equivalent non-dimensional resistance and other non-dimensional numbers
            side.R_eq_star = side.R_v_star + 1 / side.Bi
            side.modified_Bi = 1 / side.R_eq_star
            side.Pe_over_mod_Bi = cell.membrane.Pe / side.modified_Bi
            side.alpha = 1 - (1 if self.eod_parallel_to_sorption else 0) / side.Bi / side.R_eq_star

    def calculate_cathode_flux(self, cell):
        """
        Calculate the water flux at the cathode.

        Parameters
        ----------
        cell : FuelCell object
            An object representing the cell with properties related to water content and transport.

        Returns
        -------
        float
            The calculated water flux at the cathode (kmol/m²/s).
        """
        lmbd_ca = self.water_content_profile[-1, :]
        return (
            (
                cell.ca.modified_Bi * (lmbd_ca - cell.ca.est_water_content) +
                (1 if self.eod_parallel_to_sorption else 0) / cell.ca.Bi / cell.ca.R_eq_star * cell.membrane.Pe
            ) / cell.K_mb
        )

    def update_cell_side_water_fluxes(self, cell_side):
        """
        Calculate the liquid and vapor fluxes for a given side of the cell (cathode or anode).

        Parameters
        ----------
        cell_side : CellSide object
            An object representing a side of the cell with properties related to water and vapor content, transport,
            and other physical properties.

        Notes
        -----
        This function calculates the maximum vapor removal flux and then determines the liquid flux
        by comparing the water flux with the maximum vapor removal flux.
        """
        # Get saturation and vapor concentrations
        cl_sat_concentration = cell_side.cl.saturation_concentration()
        ch_vapor_concentration = cell_side.ch.vapor_concentration()

        # Calculate maximum vapor removal flux
        max_vapor_removal_flux = (cl_sat_concentration - ch_vapor_concentration) / cell_side.h2ov_transport_resistance

        # Calculate liquid flux, ensuring it is above a minimum threshold
        cell_side.liquid_flux = np.maximum(cell_side.water_flux - max_vapor_removal_flux, 1e-12)

        # Calculate vapor flux
        cell_side.vapor_flux = cell_side.water_flux - cell_side.liquid_flux


    def update_water_fluxes(self, cell):
        """
        Calculate water fluxes for both the cathode and anode sides of the cell.

        Parameters
        ----------
        cell : object
            An object representing the cell with properties related to water content, transport,
            and other physical properties for cathode and anode.

        Notes
        -----
        This function calculates the water flux for both the cathode and anode sides of the cell,
        and then calculates the liquid flux for each side.
        """
        # Calculate water flux for cathode and anode
        membrane_to_cathode_flux = self.calculate_cathode_flux(cell)
        cell.ca.membrane_water_flux = membrane_to_cathode_flux
        cell.an.membrane_water_flux = -membrane_to_cathode_flux
        cell.ca.water_flux = cell.h2o_production + membrane_to_cathode_flux
        cell.an.water_flux = -membrane_to_cathode_flux
        
        # Calculate liquid flux for both cathode and anode sides
        self.update_cell_side_water_fluxes(cell.ca)
        self.update_cell_side_water_fluxes(cell.an)

    def solve_water_balance(self, cell):
        """
        Calculate and update the water balance properties in the cell.

        Parameters
        ----------
        cell : FuelCell object
            An object representing the cell with properties related to water content, transport,
            membrane properties, and other physical and structural properties.

        Notes
        -----
        The water balance calculation is an extension of the work of Ferrara et al. (2018) accounting 
        for non-equilibrium conditions at the membrane interfaces. Gas transport resistances are calculated
        supposing that there is no liquid water present. 

        The function performs the following steps:
        1. Calculates the water absorption coefficient and diffusivity.
        2. Computes a ratio involving membrane thickness and water diffusivity.
        3. Calculates the Peclet number and estimates equilibrium water contents.
        4. Computes non-dimensional water vapor resistance, Biot number, and other parameters.
        5. Computes the water content profile and sets water contents in the cell.
        """
        # Calculate membrane water absorption and diffusivity
        self.absorption_coefficient = self.calculate_water_absorption_coefficient(cell.membrane.temperature)
        self.water_diffusivity = self.calculate_water_diffusivity(cell.membrane.temperature)

        # Calculate ratio between membrane thickness and water diffusivity x concentration, to be used later
        cell.K_mb = cell.membrane.dry_thickness / (self.water_diffusivity * cell.membrane.dry_concentration)

        # Calculate Peclet number and exponentials
        
        self.estimate_equilibrium_water_contents(cell)
        self.update_non_dimensional_parameters(cell)

        # Calculate water profile
        self.update_water_profile(cell)

        # Calculate water fluxes and set water contents
        self.update_water_fluxes(cell)
        self.update_water_contents(cell)
        

        return self.water_content_profile


