"""
Membrane water balance model.

:class:`MembraneWaterBalanceModel` solves for the membrane water-content profile,
electroosmotic drag, and net water flux across the membrane.

The model is based on Ferrara et al. (2018), using a 1D finite-difference
discretization of the water-diffusion equation with electroosmotic drag and
non-equilibrium sorption boundary conditions.

References
----------
Ferrara, A. et al. J. Power Sources 390, 197–207 (2018).
"""
import numpy as np
from scipy.linalg import solve
from dataclasses import dataclass, field
from marapendi.thermo.constants import GAS_CONSTANT
from marapendi.thermo.gas import GasModel
from marapendi.tools import arrhenius_term
from marapendi.thermo.water import water_molar_volume, water_dynamic_viscosity

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

    def estimate_equilibrium_water_contents(self, cell, state):
        """Estimate the CL equilibrium water content and its derivative for both sides.

        Reads gas state from *state* via GasModel so that the saturation pressure
        (already stored on each ``state.*.cl`` by :meth:`ThermalModel.set_mea_temperature`)
        is reused without being recomputed.

        Parameters
        ----------
        cell : FuelCell
            Provides the membrane physics (equilibrium water content model).
        state : CellState
            Runtime state whose side states carry pre-computed ``saturation_pressure``
            on their catalyst-layer sub-states.
        """
        membrane = cell.membrane
        membrane_temperature = state.membrane.temperature

        for side_state in state.sides:
            side_state.rh_at_cl_without_crossover = (
                (GasModel.vapor_concentration(side_state.ch)
                 + side_state.h2o_production * side_state.h2ov_transport_resistance)
                / GasModel.saturation_concentration(side_state.cl)
            )
            side_state.estimated_water_content = membrane.equilibrium_water_content(
                side_state.rh_at_cl_without_crossover, membrane_temperature, side_state.s_relax,
            )
            side_state.estimated_water_content_derivative = membrane.equilibrium_water_content_derivative(
                side_state.rh_at_cl_without_crossover, membrane_temperature, side_state.s_relax,
            )
    
    def update_water_contents(self, cell, state):
        """Set membrane and CL water contents. Writes to state; syncs to cell."""
        state.membrane.water_content = np.mean(self.water_content_profile, axis=0)
        cell.membrane.water_content = state.membrane.water_content

        state.ca.cl.membrane_interface_water_content = self.water_content_profile[-1, ...]
        state.an.cl.membrane_interface_water_content = self.water_content_profile[0, ...]
        cell.ca.cl.memb_interface_water_content = state.ca.cl.membrane_interface_water_content
        cell.an.cl.memb_interface_water_content = state.an.cl.membrane_interface_water_content

        for side, side_state in ((cell.ca, state.ca), (cell.an, state.an)):
            side_state.cl.eq_water_content = (
                side_state.cl.membrane_interface_water_content
                - side_state.membrane_water_flux / self.absorption_coefficient / cell.membrane.dry_concentration
            )
            side.cl.eq_water_content = side_state.cl.eq_water_content

    
    def update_water_profile(self, state):
        """Calculate the water content profile across the membrane. Reads from *state*."""
        Pe   = state.membrane.peclet_number
        ePe  = state.membrane.ePe
        ePexi = state.membrane.ePexi

        an_alpha = state.an.alpha
        ca_alpha = state.ca.alpha
        an_pomb  = state.an.peclet_over_modified_biot
        ca_pomb  = state.ca.peclet_over_modified_biot
        an_ewc   = state.an.estimated_water_content
        ca_ewc   = state.ca.estimated_water_content

        denominator = (
            (ePe - 1) * (
                1 + an_alpha * an_pomb - ca_alpha * ca_pomb
                - ca_alpha * an_alpha * an_pomb * ca_pomb
            )
            + an_pomb + ePe * ca_pomb
            + an_pomb * ca_pomb * (ePe * an_alpha - ca_alpha)
        )

        self.water_content_profile = (
            (
                an_ewc * ((ePe - ePexi) * (1 - ca_alpha * ca_pomb) + ePe * ca_pomb)
                + ca_ewc * ((ePexi - 1) * (1 + an_alpha * an_pomb) + an_pomb)
            ) / denominator
        )
        self.water_content_derivative_profile = (
            (
                an_ewc * (ePexi * Pe * (ca_alpha * ca_pomb - 1))
                + ca_ewc * (ePexi * Pe * (1 + an_alpha * an_pomb))
            ) / denominator
        )

    def update_non_dimensional_parameters(self, cell, state):
        """
        Calculate various non-dimensional parameters related to water transport and equilibrium in a cell.
        Reads runtime values from *state*; writes results to *state* and syncs to *cell* for
        methods that still read from the component tree.
        """
        state.membrane.peclet_number = self.calculate_peclet_number(
            state.membrane.temperature, state.current_density, cell.membrane, self.water_diffusivity)

        state.membrane.xi = np.linspace(0, np.ones_like(state.current_density), 10)
        state.membrane.ePexi = np.exp(state.membrane.xi * state.membrane.peclet_number)
        state.membrane.ePe = np.exp(state.membrane.peclet_number)

        # Sync to cell.membrane for methods that still read from the component.
        cell.membrane.peclet_number = state.membrane.peclet_number
        cell.membrane.xi = state.membrane.xi
        cell.membrane.ePexi = state.membrane.ePexi
        cell.membrane.ePe = state.membrane.ePe

        for side, side_state in ((cell.ca, state.ca), (cell.an, state.an)):
            side_state.biot_number = self.calculate_biot_number(
                self.absorption_coefficient /
                (side_state.estimated_water_content_derivative if self.sorption_activity_driving_force else 1),
                cell.membrane, self.water_diffusivity
            )
            R_v_star = (
                side_state.h2ov_transport_resistance
                / (GasModel.saturation_concentration(side_state.cl) * state.membrane.water_diffusion_resistance)
                * side_state.estimated_water_content_derivative
            )
            R_eq_star = R_v_star + 1 / side_state.biot_number
            side_state.peclet_over_modified_biot = state.membrane.peclet_number / (1 / R_eq_star)
            side_state.alpha = 1 - (1 if self.eod_parallel_to_sorption else 0) / side_state.biot_number / R_eq_star

            # Sync to cell side for methods that still read from the component.
            side.biot_number = side_state.biot_number
            side.peclet_over_modified_biot = side_state.peclet_over_modified_biot
            side.alpha = side_state.alpha
            side.h2ov_transport_resistance = side_state.h2ov_transport_resistance
            side.estimated_water_content_derivative = side_state.estimated_water_content_derivative
            side.estimated_water_content = side_state.estimated_water_content

    def calculate_cathode_membrane_flux(self, state):
        """Water flux at the cathode/membrane interface (kmol/m²/s)."""
        nd_flux = (
            -self.water_content_derivative_profile[-1, ...]
            + state.membrane.peclet_number * self.water_content_profile[-1, ...]
        )
        return nd_flux / state.membrane.water_diffusion_resistance

    def calculate_anode_membrane_flux(self, state):
        """Water flux at the anode/membrane interface (kmol/m²/s)."""
        nd_flux = (
            self.water_content_derivative_profile[0, ...]
            - state.membrane.peclet_number * self.water_content_profile[0, ...]
        )
        return nd_flux / state.membrane.water_diffusion_resistance

    def update_cell_side_water_fluxes(self, cell_side, side_state):
        """Compute liquid and vapor fluxes for one cell side from state gas data."""
        max_vapor_removal_flux = (
            (GasModel.saturation_concentration(side_state.cl) - GasModel.vapor_concentration(side_state.ch))
            / side_state.h2ov_transport_resistance
        )
        side_state.liquid_flux = np.maximum(side_state.water_flux - max_vapor_removal_flux, 0)
        side_state.vapor_flux = side_state.water_flux - side_state.liquid_flux
        # Sync to cell side for backward-compatible readers.
        cell_side.liquid_flux = side_state.liquid_flux
        cell_side.vapor_flux = side_state.vapor_flux

    def update_water_fluxes(self, cell, state, dynamic=False):
        """Calculate water fluxes for both sides. Reads/writes state; syncs to cell."""
        state.ca.membrane_water_flux = self.calculate_cathode_membrane_flux(state)
        state.an.membrane_water_flux = self.calculate_anode_membrane_flux(state)
        state.ca.water_flux = state.ca.h2o_production + state.ca.membrane_water_flux
        state.an.water_flux = state.an.h2o_production + state.an.membrane_water_flux

        cell.ca.membrane_water_flux = state.ca.membrane_water_flux
        cell.an.membrane_water_flux = state.an.membrane_water_flux
        cell.ca.water_flux = state.ca.water_flux
        cell.an.water_flux = state.an.water_flux

        self.update_cell_side_water_fluxes(cell.ca, state.ca)
        self.update_cell_side_water_fluxes(cell.an, state.an)

        if dynamic:
            self.membrane_diffusion_flux = (
                -self.water_content_derivative_profile / state.membrane.water_diffusion_resistance
            )
            self.membrane_eod_flux = state.membrane.peclet_number * self.water_content_profile
            self.membrane_water_flux = self.membrane_diffusion_flux + self.membrane_eod_flux

            self.membrane_water_net_flux = -np.diff(
                self.membrane_diffusion_flux, prepend=0, append=0, axis=0,
            )
            self.membrane_water_net_flux[0, ...] -= state.an.membrane_water_flux
            self.membrane_water_net_flux[-1, ...] -= state.ca.membrane_water_flux
    
    def calculate_water_saturation(self, cell_side, side_state, calculate_cl_saturation=True) -> None:
        """Compute and update water saturation in each porous layer.

        Parameters
        ----------
        cell_side : FuelCellSide
            Component side — provides static layer parameters (contact angle,
            two-phase transport model, ``has_gdl``, ``has_mpl``).
        side_state : CellSideState
            Runtime state for this side — ``liquid_flux`` and ``gas_flux`` must
            already be set; saturation fields are written to each layer state.
        calculate_cl_saturation : bool, optional
            When ``False``, the catalyst-layer saturation is not recomputed (used
            when the CL saturation is prescribed externally, e.g. in transient
            models). Default is ``True``.
        """
        for layer, ls in zip(cell_side.porous_layers,side_state.porous_layers):
            if layer is not cell_side.cl or calculate_cl_saturation: 
                ls.non_wetting_flux = side_state.liquid_flux if layer.contact_angle > 90 else side_state.gas_flux
                    
        if cell_side.has_gdl:
            cell_side.gdl.two_phase_transport_model.calculate_non_wetting_saturation(
                cell_side.gdl, side_state.gdl,
                upstream_capillary_pressure=np.zeros_like(side_state.gdl.non_wetting_flux))
            if cell_side.has_mpl:
                cell_side.mpl.two_phase_transport_model.calculate_non_wetting_saturation(
                    cell_side.mpl, side_state.mpl,
                    upstream_capillary_pressure=side_state.gdl.downstream_capillary_pressure)
                if calculate_cl_saturation: 
                    cell_side.cl.two_phase_transport_model.calculate_non_wetting_saturation(
                        cell_side.cl, side_state.cl,
                        upstream_capillary_pressure=side_state.mpl.downstream_capillary_pressure)
            else:
                if calculate_cl_saturation: 
                    cell_side.cl.two_phase_transport_model.calculate_non_wetting_saturation(
                        cell_side.cl, side_state.cl,
                        upstream_capillary_pressure=side_state.gdl.downstream_capillary_pressure)
        else:
            if calculate_cl_saturation: 
                cell_side.cl.two_phase_transport_model.calculate_non_wetting_saturation(
                    cell_side.cl, side_state.cl,
                    upstream_capillary_pressure=np.zeros_like(side_state.cl.non_wetting_flux))

        for layer, ls in self._layer_pairs(cell_side, side_state):
            ls.liquid_saturation = (
                ls.non_wetting_saturation if layer.contact_angle > 90.
                else (1 - ls.non_wetting_saturation)
            )
            ls.electrolyte_saturation = ls.liquid_saturation

    @staticmethod
    def _layer_pairs(cell_side, side_state):
        """Yield ``(component_layer, layer_state)`` pairs in the same order."""
        return zip(cell_side.porous_layers, side_state.porous_layers)

    def calculate_water_transport(self, cell, state, dynamic: bool = False,
                                   gas_transport_model=None) -> None:
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
        gas_transport_model : GasTransportModel, optional
            Shared instance for computing H₂O vapor transport resistances.
            A temporary instance is created if not provided.
        """
        from .gas_transport import GasTransportModel
        _gtr = gas_transport_model if gas_transport_model is not None else GasTransportModel()
        htr_ca = _gtr.gas_transport_resistance(cell.ca, state.ca, 'h2o')
        htr_an = _gtr.gas_transport_resistance(cell.an, state.an, 'h2o')
        state.ca.h2ov_transport_resistance = htr_ca
        state.an.h2ov_transport_resistance = htr_an

        self.solve_water_balance(cell, state=state)

        # Propagate eq water contents to state for ohmic overpotential.
        for side_state, cell_side in [(state.ca, cell.ca), (state.an, cell.an)]:
            side_state.liquid_eq_water_content = getattr(cell_side, 'liquid_eq_water_content', None)
            side_state.vapor_eq_water_content  = getattr(cell_side, 'vapor_eq_water_content', None)

        if not dynamic:
            self.calculate_water_saturation(cell.ca, state.ca)
            cell.ca.cl.set_water_film_thickness(state.ca.cl.non_wetting_saturation)
            htr_ca = _gtr.gas_transport_resistance(cell.ca, state.ca, 'h2o')
            htr_an = _gtr.gas_transport_resistance(cell.an, state.an, 'h2o')
            state.ca.h2ov_transport_resistance = htr_ca
            state.an.h2ov_transport_resistance = htr_an

        for cl_comp, cl_state in [(cell.ca.cl, state.ca.cl), (cell.an.cl, state.an.cl)]:
            if cell.use_eq_water_content_for_ionomer:
                cl_state.ionomer_water_content = cl_state.eq_water_content
            else:
                cl_state.ionomer_water_content = cl_state.membrane_interface_water_content
            cl_comp.set_ionomer_wet_properties(cl_state.ionomer_water_content, cl_comp.temperature)

    def solve_water_balance(self, cell, state=None, water_profile=None, dynamic=False):
        """
        Calculate and update the water balance properties in the cell.

        Parameters
        ----------
        cell : FuelCell
            Cell object containing membrane, cathode (ca), and anode (an) components.
        state : CellState, optional
            When provided, gas reads in :meth:`estimate_equilibrium_water_contents`
            use the pre-computed saturation pressures on ``state.*.cl`` rather than
            recomputing them. Results are also written to ``state.sides``.
        water_profile : np.ndarray, optional
            Initial water content profile, required when dynamic=True.
        dynamic : bool
            If True, solves for a prescribed water profile without updating fluxes.
        """
        mem_temp = state.membrane.temperature
        self._initialize_saturation_water_contents(cell, mem_temp)
        self._initialize_interface_water_contents(cell, state, water_profile, dynamic)

        self.absorption_coefficient = cell.membrane.calculate_water_absorption_coefficient(mem_temp)
        self.water_diffusivity = cell.membrane.calculate_water_diffusivity(mem_temp)
        state.membrane.water_diffusion_resistance = (
            cell.membrane.dry_thickness / (self.water_diffusivity * cell.membrane.dry_concentration)
        )
        cell.membrane_water_diffusion_resistance = state.membrane.water_diffusion_resistance

        self.estimate_equilibrium_water_contents(cell, state)

        self.update_non_dimensional_parameters(cell, state)

        if not dynamic:
            self._solve_static_water_balance(cell, state)
        else:
            self.update_water_fluxes(cell, state, dynamic)
            self.update_water_contents(cell, state)

        return self.water_content_profile


    def _initialize_saturation_water_contents(self, cell, mem_temp):
        """Cache vapor- and liquid-equilibrium water contents at full saturation for use as boundary limits."""
        self.vapor_equilibrium_saturation_water_content = cell.membrane.equilibrium_water_content(
            rh=1., temperature=mem_temp)
        self.liquid_equilibrium_saturation_water_content = cell.membrane.liquid_equilibrium_water_content(
            temperature=mem_temp)

    def _initialize_interface_water_contents(self, cell, state, water_profile, dynamic):
        """Set membrane interface water contents from profile (dynamic) or zero (static)."""
        if dynamic:
            self.water_content_profile = water_profile
            cell.ca.cl.membrane_interface_water_content = self.water_content_profile[-1, ...]
            cell.an.cl.membrane_interface_water_content = self.water_content_profile[0, ...]
            state.ca.cl.membrane_interface_water_content = self.water_content_profile[-1, ...]
            state.an.cl.membrane_interface_water_content = self.water_content_profile[0, ...]
        else:
            for side, side_state in ((cell.ca, state.ca), (cell.an, state.an)):
                side.cl.membrane_interface_water_content = 0
                side_state.cl.membrane_interface_water_content = 0

    def _solve_static_water_balance(self, cell, state):
        """Solve static water balance, branching into liquid-equilibrated treatment if needed."""
        self.update_water_profile(state)
        self.update_water_fluxes(cell, state, dynamic=False)
        self.update_water_contents(cell, state)

        state.ca.is_liquid_equilibrated = state.ca.liquid_flux > 0

        if np.any(state.ca.is_liquid_equilibrated):
            self._solve_liquid_equilibrated_balance(cell, state)
        else:
            self._store_vapor_equilibrium_state(cell, state)
            self._store_liquid_equilibrium_state(cell, state)

    def _solve_liquid_equilibrated_balance(self, cell, state):
        """
        Three-stage solution for liquid-equilibrated cathode locations:
        1. Apply vapor-equilibrium boundary condition and store resulting state.
        2. Apply liquid-equilibrium boundary condition and store resulting state.
        3. Blend the two states weighted by non-wetting saturation.
        """
        self._store_vapor_equilibrium_state(cell, state)
        self._apply_liquid_equilibrium_condition(cell, state)
        self._store_liquid_equilibrium_state(cell, state)

        state.ca.cl.non_wetting_saturation[state.ca.is_liquid_equilibrated] = (
            self._compute_non_wetting_saturation(cell, state)
        )
        cell.ca.cl.non_wetting_saturation = state.ca.cl.non_wetting_saturation

        self._blend_equilibrium_profiles(cell, state)
        self._update_blended_fluxes(cell, state)

        self.update_cell_side_water_fluxes(cell.ca, state.ca)
        self.update_cell_side_water_fluxes(cell.an, state.an)


    def _store_vapor_equilibrium_state(self, cell, state):
        """Store water fluxes and profiles after applying the vapor-equilibrium condition."""
        cell.ca.vapor_eq_sat_membrane_water_flux = state.ca.membrane_water_flux.copy()
        cell.ca.vapor_eq_sat_water_flux          = state.ca.water_flux.copy()
        cell.ca.vapor_eq_sat_liquid_flux         = state.ca.liquid_flux.copy()
        cell.ca.vapor_eq_water_content           = state.ca.cl.eq_water_content.copy()
        state.ca.vapor_eq_water_content          = cell.ca.vapor_eq_water_content
        cell.membrane.vapor_eq_sat_water_profile = self.water_content_profile.copy()
        cell.membrane.vapor_eq_sat_water_derivative_profile = self.water_content_derivative_profile.copy()

    def _apply_liquid_equilibrium_condition(self, cell, state):
        """Impose liquid-equilibrium water content at liquid-equilibrated cathode locations."""
        mask = state.ca.is_liquid_equilibrated
        state.ca.peclet_over_modified_biot = np.where(
            mask,
            state.membrane.peclet_number / state.ca.biot_number / 100,
            state.ca.peclet_over_modified_biot,
        )
        cell.ca.peclet_over_modified_biot = state.ca.peclet_over_modified_biot
        state.ca.estimated_water_content = np.where(
            mask,
            self.liquid_equilibrium_saturation_water_content,
            state.ca.estimated_water_content,
        )
        cell.ca.estimated_water_content = state.ca.estimated_water_content
        self.update_water_profile(state)
        self.update_water_fluxes(cell, state, dynamic=False)
        self.update_water_contents(cell, state)

    def _store_liquid_equilibrium_state(self, cell, state):
        """Store water fluxes and profiles after applying the liquid-equilibrium condition."""
        cell.ca.liquid_eq_sat_membrane_water_flux = state.ca.membrane_water_flux.copy()
        cell.ca.liquid_eq_sat_water_flux          = state.ca.water_flux.copy()
        cell.ca.liquid_eq_sat_liquid_flux         = state.ca.liquid_flux.copy()
        cell.ca.liquid_eq_water_content           = state.ca.cl.eq_water_content.copy()
        state.ca.liquid_eq_water_content          = cell.ca.liquid_eq_water_content
        cell.membrane.liquid_eq_sat_water_profile = self.water_content_profile.copy()
        cell.membrane.liquid_eq_sat_water_derivative_profile = self.water_content_derivative_profile.copy()

    def _compute_non_wetting_saturation(self, cell, state, n_iter=5):
        """Estimate non-wetting saturation at liquid-equilibrated cathode locations."""
        mask = state.ca.is_liquid_equilibrated
        params = self._assemble_saturation_equation_params(cell, state, mask)
        return self._halley_solve_saturation(*params, n_iter)

    def _assemble_saturation_equation_params(self, cell, state, liquid_equilibrated_mask):
        """Assemble coefficients for the non-wetting saturation equation."""
        cathode = cell.ca
        gdl_permeability_exponent = cathode.gdl.relative_permeability_exponent
        cl_J_function_exponent    = cathode.cl.two_phase_transport_model.J_function_exponent
        gdl_J_function_exponent   = cathode.gdl.two_phase_transport_model.J_function_exponent

        combined_flux_exponent = (
            gdl_permeability_exponent * cl_J_function_exponent / gdl_J_function_exponent
            + cl_J_function_exponent
        )
        cl_to_gdl_capillary_pressure_ratio = (
            state.ca.cl.breakthrough_pressure / state.ca.gdl.breakthrough_pressure
        )
        modified_abs_permeability = (
            np.ones_like(state.current_density)
            * 1. / state.ca.gdl.saturation_flow_resistance
            * gdl_J_function_exponent / (gdl_permeability_exponent + gdl_J_function_exponent)
            * cl_to_gdl_capillary_pressure_ratio ** (gdl_permeability_exponent / gdl_J_function_exponent + 1)
        )[liquid_equilibrated_mask]

        membrane_flux_difference = (
            cathode.vapor_eq_sat_membrane_water_flux - cathode.liquid_eq_sat_membrane_water_flux
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


    def membrane_water_rate_of_change(self, cell, n_membrane_mesh: int) -> np.ndarray:
        """Compute dλ/dt for each membrane mesh node.

        Parameters
        ----------
        cell : FuelCell
            Provides ``cell.membrane.surface_concentration``.
        n_membrane_mesh : int
            Number of membrane mesh nodes.
        """
        return self.membrane_water_net_flux / (cell.membrane.surface_concentration / n_membrane_mesh)

    def relaxation_rate_of_change(self, cell) -> np.ndarray:
        """Compute ds_relax/dt for the cathode and anode relaxation state.

        Parameters
        ----------
        cell : FuelCell
            Provides membrane relaxation parameters and MEA temperature.
        """
        dxdt = []
        for side in (cell.ca, cell.an):
            side.t_relax = (
                cell.membrane.relaxation_time_constant
                * np.exp(
                    cell.membrane.relaxation_time_activation_energy
                    / GAS_CONSTANT / cell.mea_temperature
                )
                / np.where(side.membrane_water_flux < 0, 1., 2.)
            )
            dxdt += [
                -(side.s_relax - cell.membrane.uptake_relaxed_fraction_constant * side.est_water_content)
                / side.t_relax
            ]
        return np.array(dxdt)

    def saturation_rate_of_change(self, cell) -> np.ndarray:
        """Compute d(saturation)/dt for each porous layer on anode and cathode.

        Parameters
        ----------
        cell : FuelCell
            Provides the component tree (porous layers, side geometry) and
            ``mea_water_molar_volume`` for the unit conversion.
        """
        dsdt = []
        for side in (cell.an, cell.ca):
            for layer in side.porous_layers:
                layer.flow_resistance_with_rel_permeability = (
                    layer.saturation_flow_resistance
                    * layer.breakthrough_pressure
                    / np.maximum(layer.non_wetting_saturation, 1e-1)
                    ** (layer.relative_permeability_exponent + 2)
                )
                layer.capillary_pressure = layer.capillary_pressure_from_saturation(
                    layer.non_wetting_saturation
                )

            if side.has_mpl:
                side.cl_to_mpl_liquid_flux = (
                    2 / (side.cl.flow_resistance_with_rel_permeability + side.mpl.flow_resistance_with_rel_permeability)
                    * (side.cl.capillary_pressure - side.mpl.capillary_pressure)
                )
                side.mpl_to_gdl_liquid_flux = (
                    2 / (side.mpl.flow_resistance_with_rel_permeability + side.gdl.flow_resistance_with_rel_permeability)
                    * (side.mpl.capillary_pressure - side.gdl.capillary_pressure)
                )
            else:
                side.cl_to_gdl_liquid_flux = (
                    2 / (side.cl.flow_resistance_with_rel_permeability + side.gdl.flow_resistance_with_rel_permeability)
                    * (side.cl.capillary_pressure - side.gdl.capillary_pressure)
                )
            side.gdl_to_ch_liquid_flux = (
                2 / side.gdl.flow_resistance_with_rel_permeability
                * (side.gdl.capillary_pressure - 0)
            )

            if side.has_mpl:
                side.cl.liquid_balance  = side.liquid_flux - side.cl_to_mpl_liquid_flux
                side.mpl.liquid_balance = side.cl_to_mpl_liquid_flux - side.mpl_to_gdl_liquid_flux
                side.gdl.liquid_balance = side.mpl_to_gdl_liquid_flux - side.gdl_to_ch_liquid_flux
            else:
                side.cl.liquid_balance  = side.liquid_flux - side.cl_to_gdl_liquid_flux
                side.gdl.liquid_balance = side.cl_to_gdl_liquid_flux - side.gdl_to_ch_liquid_flux

            for layer in side.porous_layers:
                dsdt.append(
                    layer.liquid_balance / (layer.porosity * layer.thickness)
                    * cell.mea_water_molar_volume
                )
        return np.array(dsdt)

    def _blend_equilibrium_profiles(self, cell, state):
        """Linearly interpolate water content profiles between vapor- and liquid-equilibrium states."""
        liquid_saturation = state.ca.cl.liquid_saturation
        self.water_content_profile = (
            cell.membrane.vapor_eq_sat_water_profile * (1 - liquid_saturation)
            + cell.membrane.liquid_eq_sat_water_profile * liquid_saturation
        )
        self.water_content_derivative_profile = (
            cell.membrane.vapor_eq_sat_water_derivative_profile * (1 - liquid_saturation)
            + cell.membrane.liquid_eq_sat_water_derivative_profile * liquid_saturation
        )
        state.ca.cl.eq_water_content = (
            state.ca.vapor_eq_water_content * (1 - liquid_saturation)
            + state.ca.liquid_eq_water_content * liquid_saturation
        )
        cell.ca.cl.eq_water_content = state.ca.cl.eq_water_content
        state.membrane.water_content = np.mean(self.water_content_profile, axis=0)
        cell.membrane.water_content = state.membrane.water_content

    def _update_blended_fluxes(self, cell, state):
        """Update membrane and total water fluxes using the blended non-wetting saturation."""
        membrane_flux_difference = (
            cell.ca.vapor_eq_sat_membrane_water_flux - cell.ca.liquid_eq_sat_membrane_water_flux
        )
        non_wetting_saturation = state.ca.cl.non_wetting_saturation

        state.ca.membrane_water_flux = (
            cell.ca.vapor_eq_sat_membrane_water_flux
            - membrane_flux_difference * non_wetting_saturation
        )
        state.an.membrane_water_flux = -state.ca.membrane_water_flux
        state.ca.water_flux = state.ca.h2o_production + state.ca.membrane_water_flux
        state.an.water_flux = state.an.h2o_production + state.an.membrane_water_flux
        cell.ca.membrane_water_flux = state.ca.membrane_water_flux
        cell.an.membrane_water_flux = state.an.membrane_water_flux
        cell.ca.water_flux = state.ca.water_flux
        cell.an.water_flux = state.an.water_flux
