import numpy as np
from scipy.linalg import solve
from dataclasses import dataclass, field
from marapendi.tools import arrhenius_term
from marapendi.water import water_molar_volume, water_dynamic_viscosity

@dataclass
class MembraneWaterBalanceModel:
    """
    A class representing a membrane water balance model with various parameters related to water diffusivity and absorption.

    Attributes
    ----------
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
    sorption_activity_driving_force: bool = False
    eod_parallel_to_sorption: bool = False                            


    
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
        return (membrane.calculate_electroosmotic_drag_speed(temperature, current_density) *
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
            side.rh_at_cl_without_crossover = (
                (side.ch.vapor_concentration() + side.h2o_production * side.h2ov_transport_resistance)
                / side.cl.saturation_concentration()
            )

            # Get estimated values for equilibrium water content and its derivative in CL 
            
            side.estimated_water_content = cell.membrane.equilibrium_water_content(
                    side.rh_at_cl_without_crossover, 
                    cell.membrane.temperature, 
                    side.s_relax
                )
            side.estimated_water_content_derivative = cell.membrane.equilibrium_water_content_derivative(
                    side.rh_at_cl_without_crossover, 
                    cell.membrane.temperature, 
                    side.s_relax
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
        cell.ca.cl.memb_interface_water_content = self.water_content_profile[-1,...]
        cell.an.cl.memb_interface_water_content = self.water_content_profile[0,...]

        # Calculate equilibrium water contents at the CL
        for side in (cell.ca, cell.an):
            side.cl.eq_water_content = (side.cl.memb_interface_water_content -
                                        side.membrane_water_flux / self.absorption_coefficient / cell.membrane.dry_concentration)
                                            
    
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
        Pe = cell.membrane.peclet_number 
        ePe = cell.membrane.ePe
        ePexi = cell.membrane.ePexi

        denominator = (
                (ePe - 1) * (
                    1 + cell.an.alpha * cell.an.peclet_over_modified_biot - cell.ca.alpha * cell.ca.peclet_over_modified_biot +
                    -cell.ca.alpha * cell.an.alpha * cell.an.peclet_over_modified_biot * cell.ca.peclet_over_modified_biot
                ) +
                cell.an.peclet_over_modified_biot + ePe * cell.ca.peclet_over_modified_biot +
                cell.an.peclet_over_modified_biot * cell.ca.peclet_over_modified_biot * (ePe * cell.an.alpha - cell.ca.alpha)
            )
        
        # Calculate the water content profile using a detailed mathematical formula
        self.water_content_profile = (
            (
                cell.an.estimated_water_content * ((ePe - ePexi) * (1 - cell.ca.alpha * cell.ca.peclet_over_modified_biot) + ePe * cell.ca.peclet_over_modified_biot) +
                cell.ca.estimated_water_content * ((ePexi - 1) * (1 + cell.an.alpha * cell.an.peclet_over_modified_biot) + cell.an.peclet_over_modified_biot)
            ) / denominator
        )
        self.water_content_derivative_profile = (
            (
                cell.an.estimated_water_content * (ePexi * Pe * (cell.ca.alpha * cell.ca.peclet_over_modified_biot - 1)) +
                cell.ca.estimated_water_content * (ePexi * Pe * (1 + cell.an.alpha * cell.an.peclet_over_modified_biot) )
            ) / denominator 
        )
        
        cell.an.a = Pe * ePe * (1 + (1- cell.ca.alpha) * cell.ca.peclet_over_modified_biot) / denominator
        cell.ca.a = - Pe * (1 - (1 - cell.an.alpha) * cell.an.peclet_over_modified_biot ) / denominator

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
        cell.membrane.peclet_number = self.calculate_peclet_number(cell.membrane.temperature, cell.current_density, cell.membrane, self.water_diffusivity)

        # Generate evenly spaced points over the interval
        cell.membrane.xi = np.linspace(0, np.ones_like(cell.current_density), 10)
        cell.membrane.ePexi = np.exp(cell.membrane.xi * cell.membrane.peclet_number)
        cell.membrane.ePe = np.exp(cell.membrane.peclet_number)

        for side in (cell.ca, cell.an):
            # Calculate non-dimensional water vapor resistance based on a water content driving force
            side.R_v_star = side.h2ov_transport_resistance / (side.cl.saturation_concentration() * cell.membrane_water_diffusion_resistance) * side.estimated_water_content_derivative

            # Calculate Biot number
            side.biot_number = self.calculate_biot_number(
                self.absorption_coefficient / 
                (side.estimated_water_content_derivative if self.sorption_activity_driving_force else 1),
                cell.membrane, self.water_diffusivity
            )

            # Calculate equivalent non-dimensional resistance and other non-dimensional numbers
            side.R_eq_star = side.R_v_star + 1 / side.biot_number
            side.modified_Bi = 1 / side.R_eq_star
            side.peclet_over_modified_biot = cell.membrane.peclet_number / side.modified_Bi
            side.alpha = 1 - (1 if self.eod_parallel_to_sorption else 0) / side.biot_number / side.R_eq_star

    def calculate_cathode_membrane_flux(self, cell):
        """
        Calculate the water flux at the cathode/membrane interface.

        Parameters
        ----------
        cell : FuelCell object
            An object representing the cell with properties related to water content and transport.

        Returns
        -------
        float
            The calculated water flux at the cathode (kmol/m²/s).
        """
        
        non_dimensional_membrane_water_flux = - self.water_content_derivative_profile[-1,...] + cell.membrane.peclet_number * self.water_content_profile[-1, ...]
   
        return non_dimensional_membrane_water_flux / cell.membrane_water_diffusion_resistance

    def calculate_anode_membrane_flux(self, cell):
        """
        Calculate the water flux at the anode/membrane interface.

        Parameters
        ----------
        cell : FuelCell object
            An object representing the cell with properties related to water content and transport.

        Returns
        -------
        float
            The calculated water flux at the anode (kmol/m²/s).
        """
        non_dimensional_membrane_water_flux = self.water_content_derivative_profile[0,...] - cell.membrane.peclet_number * self.water_content_profile[0,...]
        return non_dimensional_membrane_water_flux / cell.membrane_water_diffusion_resistance 

    def update_cell_side_water_fluxes(self, cell_side, dynamic):
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

        # Calculate maximum vapor removal flux
        max_vapor_removal_flux = cell_side.max_water_vapor_removal()

        # Calculate liquid flux, ensuring it is above a minimum threshold
        cell_side.liquid_flux = np.maximum(cell_side.water_flux - max_vapor_removal_flux, 0)

        # Calculate vapor flux
        cell_side.vapor_flux = cell_side.water_flux - cell_side.liquid_flux

    def update_water_fluxes(self, cell, dynamic=False):
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
        cell.ca.membrane_water_flux = self.calculate_cathode_membrane_flux(cell)
        cell.an.membrane_water_flux = self.calculate_anode_membrane_flux(cell)
        cell.ca.water_flux = cell.ca.h2o_production + cell.ca.membrane_water_flux
        cell.an.water_flux = cell.an.h2o_production + cell.an.membrane_water_flux
        
        # Calculate liquid flux for both cathode and anode sides
        self.update_cell_side_water_fluxes(cell.ca, dynamic)
        self.update_cell_side_water_fluxes(cell.an, dynamic)

        if dynamic: 
            self.membrane_diffusion_flux = (- self.water_content_derivative_profile / cell.membrane_water_diffusion_resistance)
            self.membrane_eod_flux = cell.membrane.peclet_number * self.water_content_profile
            self.membrane_water_flux = self.membrane_diffusion_flux + self.membrane_eod_flux
            
            self.membrane_water_net_flux = -np.diff(self.membrane_diffusion_flux,
                                                    prepend=0,
                                                    append=0, axis=0)

            self.membrane_water_net_flux[0,...] -= cell.an.membrane_water_flux
            self.membrane_water_net_flux[-1,...] -= cell.ca.membrane_water_flux
    
    def calculate_water_saturation(self, cell_side, side_state) -> None:
        """Compute and update water saturation in each porous layer.

        Parameters
        ----------
        cell_side : FuelCellSide
            Component side — provides static layer parameters (contact angle,
            two-phase transport model, ``has_gdl``, ``has_mpl``).
        side_state : CellSideState
            Runtime state for this side — ``liquid_flux`` and ``gas_flux`` must
            already be set; saturation fields are written here and also synced
            back to the component objects for backward compatibility.
        """
        for layer, ls in self._layer_pairs(cell_side, side_state):
            ls.non_wetting_flux = side_state.liquid_flux if layer.contact_angle > 90 else side_state.gas_flux
            ls.downstream_saturation = np.zeros_like(ls.non_wetting_flux)
            ls.upstream_saturation = np.zeros_like(ls.non_wetting_flux)
            ls.non_wetting_saturation = np.zeros_like(ls.non_wetting_flux)
            ls.downstream_capillary_pressure = np.zeros_like(ls.non_wetting_flux)
            # Keep component in sync so two_phase_transport_model reads see right values.
            layer.non_wetting_flux = ls.non_wetting_flux
            layer.downstream_saturation = ls.downstream_saturation
            layer.upstream_saturation = ls.upstream_saturation
            layer.non_wetting_saturation = ls.non_wetting_saturation
            layer.downstream_capillary_pressure = ls.downstream_capillary_pressure

        if cell_side.has_gdl:
            cell_side.gdl.two_phase_transport_model.calculate_non_wetting_saturation(
                cell_side.gdl, cell_side.gdl.non_wetting_flux,
                upstream_capillary_pressure=np.zeros_like(cell_side.cl.non_wetting_flux))
            if cell_side.has_mpl:
                cell_side.mpl.two_phase_transport_model.calculate_non_wetting_saturation(
                    cell_side.mpl, cell_side.mpl.non_wetting_flux,
                    upstream_capillary_pressure=cell_side.gdl.downstream_capillary_pressure)
                cell_side.cl.two_phase_transport_model.calculate_non_wetting_saturation(
                    cell_side.cl, side_state.liquid_flux,
                    upstream_capillary_pressure=cell_side.mpl.downstream_capillary_pressure)
            else:
                cell_side.cl.two_phase_transport_model.calculate_non_wetting_saturation(
                    cell_side.cl, side_state.liquid_flux,
                    upstream_capillary_pressure=cell_side.gdl.downstream_capillary_pressure)
        else:
            cell_side.cl.two_phase_transport_model.calculate_non_wetting_saturation(
                cell_side.cl, cell_side.cl.non_wetting_flux,
                upstream_capillary_pressure=np.zeros_like(cell_side.cl.non_wetting_flux))

        for layer, ls in self._layer_pairs(cell_side, side_state):
            ls.liquid_saturation = (
                layer.non_wetting_saturation if layer.contact_angle > 90.
                else (1 - layer.non_wetting_saturation)
            )
            ls.electrolyte_saturation = ls.liquid_saturation
            # Sync back to component.
            layer.liquid_saturation = ls.liquid_saturation
            layer.electrolyte_saturation = ls.electrolyte_saturation

    @staticmethod
    def _layer_pairs(cell_side, side_state):
        """Yield ``(component_layer, layer_state)`` pairs in the same order."""
        pairs = [
            (cell_side.cl, side_state.cl),
            (cell_side.mpl if cell_side.has_mpl else None, side_state.mpl),
            (cell_side.gdl if cell_side.has_gdl else None, side_state.gdl),
        ]
        return [(c, s) for c, s in pairs if c is not None and s is not None]

    def calculate_water_transport(self, cell, state, dynamic: bool = False) -> None:
        """Calculate the water balance across the fuel cell.

        Updates vapor transport resistances, solves the membrane water balance,
        and (when ``dynamic=False``) recalculates liquid saturation in the cathode.

        Parameters
        ----------
        cell : FuelCell
            Component tree providing static physics and legacy state attributes.
        state : CellState
            Runtime state: h2ov transport resistances and water-flux fields are
            written here and also synced to the component tree for backward
            compatibility.
        dynamic : bool
            When ``True``, skips the liquid saturation update (used by transient
            models).
        """
        # Compute H2O vapor transport resistances and write to both state and cell.
        htr_ca = cell.ca.gas_transport_resistance('h2o')
        htr_an = cell.an.gas_transport_resistance('h2o')
        state.ca.h2ov_transport_resistance = htr_ca
        state.an.h2ov_transport_resistance = htr_an
        cell.ca.h2ov_transport_resistance = htr_ca
        cell.an.h2ov_transport_resistance = htr_an

        self.solve_water_balance(cell)

        # Sync water-flux and water-content results from component tree to state.
        state.ca.liquid_flux = cell.ca.liquid_flux
        state.ca.vapor_flux  = cell.ca.vapor_flux
        state.ca.water_flux  = cell.ca.water_flux
        state.ca.membrane_water_flux = cell.ca.membrane_water_flux
        state.an.liquid_flux = cell.an.liquid_flux
        state.an.vapor_flux  = cell.an.vapor_flux
        state.an.water_flux  = cell.an.water_flux
        state.an.membrane_water_flux = cell.an.membrane_water_flux
        state.membrane.water_content = cell.membrane.water_content
        # Equilibrium water contents (used by ohmic overpotential calculation).
        for side_state, cell_side in [(state.ca, cell.ca), (state.an, cell.an)]:
            side_state.liquid_eq_water_content = getattr(cell_side, 'liquid_eq_water_content', None)
            side_state.vapor_eq_water_content  = getattr(cell_side, 'vapor_eq_water_content', None)
        # CL interface water content.
        state.ca.cl.eq_water_content = cell.ca.cl.eq_water_content
        state.an.cl.eq_water_content = cell.an.cl.eq_water_content
        state.ca.cl.membrane_interface_water_content = getattr(cell.ca.cl, 'memb_interface_water_content', None)
        state.an.cl.membrane_interface_water_content = getattr(cell.an.cl, 'memb_interface_water_content', None)

        if not dynamic:
            cell.ca.calculate_equivalent_flow_resistance()
            state.ca.gas_flux = cell.ca.gas_flux
            self.calculate_water_saturation(cell.ca, state.ca)
            # set_water_film_thickness uses the component saturation value.
            cell.ca.cl.set_water_film_thickness(cell.ca.cl.non_wetting_saturation)
            state.ca.cl.water_film_thickness = cell.ca.cl.non_wetting_saturation

            htr_ca = cell.ca.gas_transport_resistance('h2o')
            htr_an = cell.an.gas_transport_resistance('h2o')
            state.ca.h2ov_transport_resistance = htr_ca
            state.an.h2ov_transport_resistance = htr_an
            cell.ca.h2ov_transport_resistance = htr_ca
            cell.an.h2ov_transport_resistance = htr_an

        for cl_comp, cl_state in [(cell.ca.cl, state.ca.cl), (cell.an.cl, state.an.cl)]:
            if cell.use_eq_water_content_for_ionomer:
                cl_state.ionomer_water_content = cl_comp.eq_water_content
            else:
                cl_state.ionomer_water_content = cl_comp.memb_interface_water_content
            cl_comp.ionomer_water_content = cl_state.ionomer_water_content
            cl_comp.set_ionomer_wet_properties(cl_comp.ionomer_water_content, cl_comp.temperature)

    def solve_water_balance(self, cell, water_profile=None, dynamic=False):
        """
        Calculate and update the water balance properties in the cell.

        Parameters
        ----------
        cell : FuelCell
            Cell object containing membrane, cathode (ca), and anode (an) components.
        water_profile : np.ndarray, optional
            Initial water content profile, required when dynamic=True.
        dynamic : bool
            If True, solves for a prescribed water profile without updating fluxes.
        """
        self._initialize_saturation_water_contents(cell)
        self._initialize_interface_water_contents(cell, water_profile, dynamic)

        self.absorption_coefficient = cell.membrane.calculate_water_absorption_coefficient(
            cell.membrane.temperature)
        self.water_diffusivity = cell.membrane.calculate_water_diffusivity(
            cell.membrane.temperature)
        cell.membrane_water_diffusion_resistance = (
            cell.membrane.dry_thickness
            / (self.water_diffusivity * cell.membrane.dry_concentration)
        )
        
        self.estimate_equilibrium_water_contents(cell)
        self.update_non_dimensional_parameters(cell)
        

        if not dynamic:
            self._solve_static_water_balance(cell)
            
        else:
            self.update_water_fluxes(cell, dynamic)
            self.update_water_contents(cell)

        
        return self.water_content_profile


    def _initialize_saturation_water_contents(self, cell):
        """Cache vapor- and liquid-equilibrium water contents at full saturation for use as boundary limits."""
        self.vapor_equilibrium_saturation_water_content = cell.membrane.equilibrium_water_content(
            rh=1., temperature=cell.membrane.temperature)
        self.liquid_equilibrium_saturation_water_content = cell.membrane.liquid_equilibrium_water_content(
            temperature=cell.membrane.temperature)


    def _initialize_interface_water_contents(self, cell, water_profile, dynamic):
        """Set membrane interface water contents from profile (dynamic) or zero (static)."""
        if dynamic:
            self.water_content_profile = water_profile
            cell.ca.cl.membrane_interface_water_content = self.water_content_profile[-1, ...]
            cell.an.cl.membrane_interface_water_content = self.water_content_profile[0, ...]
        else:
            for side in (cell.ca, cell.an):
                side.cl.membrane_interface_water_content = 0


    def _solve_static_water_balance(self, cell):
        """Solve static water balance, branching into liquid-equilibrated treatment if needed."""
        self.update_water_profile(cell)
        self.update_water_fluxes(cell, dynamic=False)
        self.update_water_contents(cell)

        cell.ca.is_liquid_equilibrated = cell.ca.liquid_flux > 0

        if np.any(cell.ca.is_liquid_equilibrated):
            self._solve_liquid_equilibrated_balance(cell)
        else: 
            self._store_vapor_equilibrium_state(cell)
            self._store_liquid_equilibrium_state(cell)
            
    def _solve_liquid_equilibrated_balance(self, cell):
        """
        Three-stage solution for liquid-equilibrated cathode locations:
        1. Apply vapor-equilibrium boundary condition and store resulting state.
        2. Apply liquid-equilibrium boundary condition and store resulting state.
        3. Blend the two states weighted by non-wetting saturation.
        """
        self._store_vapor_equilibrium_state(cell)

        self._apply_liquid_equilibrium_condition(cell)
        self._store_liquid_equilibrium_state(cell)

        cell.ca.cl.non_wetting_saturation[cell.ca.is_liquid_equilibrated] = (
            self._compute_non_wetting_saturation(cell)
        )

        self._blend_equilibrium_profiles(cell)
        self._update_blended_fluxes(cell)

        self.update_cell_side_water_fluxes(cell.ca, dynamic=False)
        self.update_cell_side_water_fluxes(cell.an, dynamic=False)


    def _store_vapor_equilibrium_state(self, cell):
        """Store water fluxes and profiles after applying the vapor-equilibrium condition."""
        cell.ca.vapor_eq_sat_membrane_water_flux   = cell.ca.membrane_water_flux.copy()
        cell.ca.vapor_eq_sat_water_flux            = cell.ca.water_flux.copy()
        cell.ca.vapor_eq_sat_liquid_flux           = cell.ca.liquid_flux.copy()
        cell.ca.vapor_eq_water_content              = cell.ca.cl.eq_water_content.copy()
        cell.membrane.vapor_eq_sat_water_profile         = self.water_content_profile.copy()
        cell.membrane.vapor_eq_sat_water_derivative_profile = self.water_content_derivative_profile.copy()


    def _apply_liquid_equilibrium_condition(self, cell):
        """
        Impose liquid-equilibrium water content at liquid-equilibrated cathode locations
        and recompute the water profile and fluxes.

        The Peclet/Biot ratio is divided by 100 to represent the much higher
        water activity at the membrane interface under liquid equilibrium.
        """
        liquid_equilibrated_mask = cell.ca.is_liquid_equilibrated
        cell.ca.peclet_over_modified_biot = np.where(
            liquid_equilibrated_mask,
            cell.membrane.peclet_number / cell.ca.biot_number / 100,
            cell.ca.peclet_over_modified_biot
        )
        cell.ca.estimated_water_content = np.where(
            liquid_equilibrated_mask,
            self.liquid_equilibrium_saturation_water_content,
            cell.ca.estimated_water_content
        )
        self.update_water_profile(cell)
        self.update_water_fluxes(cell, dynamic=False)
        self.update_water_contents(cell)

    def _store_liquid_equilibrium_state(self, cell):
        """Store water fluxes and profiles after applying the liquid-equilibrium condition."""
        cell.ca.liquid_eq_sat_membrane_water_flux    = cell.ca.membrane_water_flux.copy()
        cell.ca.liquid_eq_sat_water_flux             = cell.ca.water_flux.copy()
        cell.ca.liquid_eq_sat_liquid_flux            = cell.ca.liquid_flux.copy()
        cell.ca.liquid_eq_water_content              = cell.ca.cl.eq_water_content.copy()
        cell.membrane.liquid_eq_sat_water_profile          = self.water_content_profile.copy()
        cell.membrane.liquid_eq_sat_water_derivative_profile = self.water_content_derivative_profile.copy()


    def _compute_non_wetting_saturation(self, cell, n_iter=5):
        """
        Estimate non-wetting saturation at liquid-equilibrated cathode locations.
        Delegates parameter assembly and root-finding to separate methods.
        """
        liquid_equilibrated_mask = cell.ca.is_liquid_equilibrated
        modified_abs_permeability, membrane_flux_difference, vapor_eq_liquid_flux, combined_flux_exponent = (
            self._assemble_saturation_equation_params(cell, liquid_equilibrated_mask)
        )
        return self._halley_solve_saturation(
            modified_abs_permeability,
            membrane_flux_difference,
            vapor_eq_liquid_flux,
            combined_flux_exponent,
            n_iter
        )


    def _assemble_saturation_equation_params(self, cell, liquid_equilibrated_mask):
        """
        Assemble coefficients for the non-wetting saturation equation:
            modified_abs_permeability * s^combined_flux_exponent + membrane_flux_difference * s - vapor_eq_liquid_flux = 0

        Returns
        -------
        modified_abs_permeability : np.ndarray
            Effective absolute permeability combining GDL and CL transport properties.
        membrane_flux_difference : np.ndarray
            Difference in membrane water flux between vapor- and liquid-equilibrium states.
        vapor_eq_liquid_flux : np.ndarray
            Liquid water flux under vapor-equilibrium conditions.
        combined_flux_exponent : float
            Combined exponent m+n from GDL permeability and CL J-function exponents.
        """
        cathode = cell.ca
        gdl_permeability_exponent = cathode.gdl.relative_permeability_exponent
        cl_J_function_exponent    = cathode.cl.two_phase_transport_model.J_function_exponent
        gdl_J_function_exponent   = cathode.gdl.two_phase_transport_model.J_function_exponent

        combined_flux_exponent = (
            gdl_permeability_exponent * cl_J_function_exponent / gdl_J_function_exponent
            + cl_J_function_exponent
        )

        cl_to_gdl_capillary_pressure_ratio = (
            cathode.cl.capillary_pressure_J_ratio
            / cathode.gdl.capillary_pressure_J_ratio
        )

        modified_abs_permeability = (
            np.ones_like(cell.current_density)
            * 1. / cathode.gdl.saturation_flow_resistance
            * gdl_J_function_exponent / (gdl_permeability_exponent + gdl_J_function_exponent)
            * cl_to_gdl_capillary_pressure_ratio ** (gdl_permeability_exponent / gdl_J_function_exponent + 1)
        )[liquid_equilibrated_mask]

        membrane_flux_difference = (
            cathode.vapor_eq_sat_membrane_water_flux
            - cathode.liquid_eq_sat_membrane_water_flux
        )[liquid_equilibrated_mask]

        vapor_eq_liquid_flux = cathode.vapor_eq_sat_liquid_flux[liquid_equilibrated_mask]

        return modified_abs_permeability, membrane_flux_difference, vapor_eq_liquid_flux, combined_flux_exponent


    @staticmethod
    def _halley_solve_saturation(modified_abs_permeability, membrane_flux_difference,
                                vapor_eq_liquid_flux, combined_flux_exponent, n_iter=5):
        """
        Halley's method root-finder for the non-wetting saturation equation:
            f(s) = modified_abs_permeability * s^combined_flux_exponent
                + membrane_flux_difference * s - vapor_eq_liquid_flux = 0

        Initial guess of 0.1 is a reasonable for low saturations.
        Saturation is clipped to [0, 0.9] each iteration to remain in the physical range.
        """
        saturation = np.full_like(membrane_flux_difference, 0.1)
        for _ in range(n_iter):
            residual     = (modified_abs_permeability * saturation**combined_flux_exponent
                            + membrane_flux_difference * saturation
                            - vapor_eq_liquid_flux)
            first_deriv  = (combined_flux_exponent * modified_abs_permeability
                            * saturation**(combined_flux_exponent - 1)
                            + membrane_flux_difference)
            second_deriv = (combined_flux_exponent * (combined_flux_exponent - 1)
                            * modified_abs_permeability
                            * saturation**(combined_flux_exponent - 2))
            saturation   = np.clip(
                saturation - residual * first_deriv / (first_deriv**2 - 0.5 * residual * second_deriv),
                0, 0.9
            )
        return saturation


    def _blend_equilibrium_profiles(self, cell):
        """Linearly interpolate water content profiles between vapor- and liquid-equilibrium states, weighted by non-wetting saturation."""
        liquid_saturation = cell.ca.cl.liquid_saturation
        self.water_content_profile = (
            cell.membrane.vapor_eq_sat_water_profile * (1 - liquid_saturation)
            + cell.membrane.liquid_eq_sat_water_profile * liquid_saturation
        )
        self.water_content_derivative_profile = (
            cell.membrane.vapor_eq_sat_water_derivative_profile * (1 - liquid_saturation)
            + cell.membrane.liquid_eq_sat_water_derivative_profile * liquid_saturation
        )
        cell.ca.cl.eq_water_content =  (
            cell.ca.vapor_eq_water_content * (1 - liquid_saturation)
            + cell.ca.liquid_eq_water_content * liquid_saturation
        )
        cell.membrane.water_content = np.mean(self.water_content_profile, axis=0)

    def _update_blended_fluxes(self, cell):
        """Update membrane and total water fluxes using the blended non-wetting saturation."""
        membrane_flux_difference = (
            cell.ca.vapor_eq_sat_membrane_water_flux
            - cell.ca.liquid_eq_sat_membrane_water_flux
        )
        non_wetting_saturation = cell.ca.cl.non_wetting_saturation

        cell.ca.membrane_water_flux = (
            cell.ca.vapor_eq_sat_membrane_water_flux
            - membrane_flux_difference * non_wetting_saturation
        )
        cell.an.membrane_water_flux = -cell.ca.membrane_water_flux
        cell.ca.water_flux = cell.ca.h2o_production + cell.ca.membrane_water_flux
        cell.an.water_flux = cell.an.h2o_production + cell.an.membrane_water_flux

@dataclass 
class MatrixMembraneWaterBalanceModel: 
    def solve_water_balance(self, cell): 
        
        memb = cell.membrane
        memb.water_diffusion_resistance = (
            memb.dry_thickness
            / (memb.calculate_water_diffusivity(memb.temperature) * memb.dry_concentration)
        )

        liquid_to_vapor_absorption_factor = 1 

        for side in (cell.ca, cell.an): 
            cl = side.cl
            s = cl.liquid_saturation 

            cl.ionomer_absorption_coefficient = cl.ionomer.calculate_water_absorption_coefficient(
                cl.temperature) 
            
            cl.vapor_eq_water_content = cl.ionomer.equilibrium_water_content(cl.relative_humidity(), cl.temperature)
            cl.liquid_eq_water_content = cl.ionomer.liquid_equilibrium_water_content(cl.temperature)
            cl.eq_water_content = (
                (1-s) * cl.vapor_eq_water_content 
                + s * cl.liquid_eq_water_content * liquid_to_vapor_absorption_factor
            ) / (1-s + s * liquid_to_vapor_absorption_factor)
            cl.absorption_coefficient = (
                ((1-s) +  s * liquid_to_vapor_absorption_factor) 
                * cl.ionomer.calculate_water_absorption_coefficient(cl.temperature) 
            ) * cl.ionomer.dry_concentration
            cl.ionomer_water_resistance = (
                cl.thickness / 
                (
                    cl.ionomer.dry_concentration
                    * cl.ionomer.calculate_water_diffusivity(cl.temperature)
                    * cl.ionomer_vol_fraction 
                    / cl.ionomer.tortuosity(cl.ionomer_vol_fraction)
                )
            )

            side.eq_water_resistance = 0.5 * (memb.water_diffusion_resistance + cl.ionomer_water_resistance)
            side.biot_number = cl.absorption_coefficient * side.eq_water_resistance
            side.peclet_number = (
                memb.calculate_electroosmotic_drag_speed(memb.temperature, cell.current_density)
                * memb.dry_concentration * side.eq_water_resistance
            )
        
        # EOD flux goes from side 2 to side 1 
        if np.any(cell.an.peclet_number > 0): 
            side_1 = cell.ca
            side_2 = cell.an 
        else: 
            side_1 = cell.an
            side_2 = cell.ca

        resistance_ratio = side_1.eq_water_resistance / side_2.eq_water_resistance
        A = np.zeros((len(cell.current_density), 3, 3))
        b = np.zeros((len(cell.current_density), 3, 1))

        A[...,0,0] = 1
        A[...,0,1] = (resistance_ratio + np.abs(side_1.peclet_number))
        A[...,0,2] = - (1 + resistance_ratio + np.abs(side_1.peclet_number))
        A[...,1,0] = (1 + side_1.biot_number)
        A[...,1,2] = -(1 + np.abs(side_1.peclet_number))
        A[...,2,1] = (1 + np.abs(side_2.peclet_number) + side_2.biot_number)
        A[...,2,2] = -1
        
        b[...,1,0] = side_1.h2o_production * side_1.eq_water_resistance + side_1.biot_number * side_1.cl.eq_water_content
        b[...,2,0] = side_2.h2o_production * side_2.eq_water_resistance + side_2.biot_number * side_2.cl.eq_water_content
        
        x = solve(A,b)
        x = x.reshape(len(cell.current_density), 3)
        side_1.cl.ionomer_water_content = x[...,0].copy()
        memb.water_content      = x[...,2].copy()
        side_2.cl.ionomer_water_content = x[...,1].copy()

        for side in (side_1, side_2): 
            cl = side.cl
            side.water_flux = side.biot_number / side.eq_water_resistance * (cl.ionomer_water_content - cl.eq_water_content)
            
            side.membrane_water_flux = (
                (memb.water_content - cl.ionomer_water_content) + 
                np.abs(side.peclet_number) * (
                    memb.water_content if side == side_1 
                    else - cl.ionomer_water_content
                )
            ) / side.eq_water_resistance
            if side == cell.an: 
                print(side.water_flux, side.membrane_water_flux, side.membrane_water_flux + side.h2o_production)
        side_1.eod_flux = side_1.peclet_number / side_1.eq_water_resistance * memb.water_content
        side_2.eod_flux = side_2.peclet_number / side_2.eq_water_resistance * side_2.cl.ionomer_water_content

        cell.ca.diff_flux = (memb.water_content - cell.ca.cl.ionomer_water_content) / cell.ca.eq_water_resistance
        cell.an.diff_flux = (memb.water_content - cell.an.cl.ionomer_water_content) / cell.an.eq_water_resistance