
import numpy as np
from dataclasses import dataclass, field
from marapendi.thermo.constants import GAS_CONSTANT
from marapendi.thermo.gas import GasModel
from marapendi.tools import arrhenius_term
from marapendi.thermo.water import water_molar_volume, water_dynamic_viscosity
from ..cell.gas_transport import GasTransportModel


@dataclass
class MembraneWaterBalanceModel:

    n_profile_points: int = 10 
    sorption_activity_driving_force: bool = False
    eod_parallel_to_sorption: bool = False

    def calculate_peclet_number(self, eod_speed, membrane_thickness, water_diffusivity):
        """
        Calculate the Peclet number using electroosmotic drag speed, membrane thickness, and water diffusivity.

        Parameters
        ----------
        eod_speed : float
            Electroosmotic drag velocity (m/s).
        membrane_thickness : float
            Membrane dry thickness (m).
        water_diffusivity : float
            Adsorbed water diffusivity (m²/s).

        Returns
        -------
        float
            The calculated Peclet number, dimensionless.
        """
        return (eod_speed * membrane_thickness / water_diffusivity)

    def calculate_biot_number(self, absorption_coefficient, membrane_thickness, water_diffusivity):
        """
        Calculate the Biot number using absorption coefficient, membrane thickness, and water diffusivity.

        Parameters
        ----------
        absorption_coefficient : float
            Water absorption coefficient at the membrane surface (m/s).
        membrane_thickness : float
            Membrane dry thickness (m).
        water_diffusivity : float
            Adsorbed water diffusivity (m²/s).

        Returns
        -------
        float
            The calculated Biot number, dimensionless.
        """
        return absorption_coefficient * membrane_thickness / water_diffusivity


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
    

    def update_water_contents(self, state):
        """Set membrane and CL water contents. Writes to state."""
        state.membrane.water_content = np.mean(self.water_content_profile, axis=0)
        state.membrane.water_content_profile = self.water_content_profile.copy()
        
        state.ca.membrane_interface_water_content = self.water_content_profile[-1, ...]
        state.an.membrane_interface_water_content = self.water_content_profile[0, ...]
    
    def calculate_membrane_transport_properties(self, cell, state): 
        memb_state = state.membrane

        memb_state.eod_speed = cell.membrane.calculate_electroosmotic_drag_speed(
            memb_state.temperature, state.current_density)
        memb_state.absorption_coefficient = cell.membrane.calculate_water_absorption_coefficient(memb_state.temperature)
        memb_state.water_diffusivity = cell.membrane.calculate_water_diffusivity(memb_state.temperature)
        memb_state.water_diffusion_resistance = (
            cell.membrane.dry_thickness / (memb_state.water_diffusivity * cell.membrane.dry_concentration)
        )

    def update_non_dim_vapor_resistance(self, side_state, memb_state, cell): 
            return (
                side_state.h2ov_transport_resistance
                / (GasModel.saturation_concentration(side_state.cl) * memb_state.water_diffusion_resistance)
                * side_state.estimated_water_content_derivative
            )

    def update_non_dimensional_parameters(self, cell, state, pwl_interval=None):
        """
        Calculate various non-dimensional parameters related to water transport and equilibrium in a cell.
        Reads runtime values from *state*; writes results to *state* and syncs to *cell* for
        methods that still read from the component tree.
        """
        memb_state = state.membrane

        memb_state.peclet_number = self.calculate_peclet_number(
            memb_state.eod_speed, cell.membrane.dry_thickness, memb_state.water_diffusivity)

        memb_state.xi = np.linspace(0, np.ones_like(state.current_density), self.n_profile_points)
        memb_state.ePexi = np.exp(memb_state.xi * memb_state.peclet_number)
        memb_state.ePe = np.exp(memb_state.peclet_number)

        for side_state in state.sides:
            side_state.biot_number = self.calculate_biot_number(
                memb_state.absorption_coefficient /
                (side_state.estimated_water_content_derivative if self.sorption_activity_driving_force else 1),
                cell.membrane.dry_thickness, memb_state.water_diffusivity
            )
            side_state.non_dim_vapor_resistance = self.update_non_dim_vapor_resistance(side_state, memb_state, cell)
            side_state.non_dim_equiv_resistance = side_state.non_dim_vapor_resistance + 1 / side_state.biot_number
            side_state.peclet_over_modified_biot = memb_state.peclet_number / (1 / side_state.non_dim_equiv_resistance)
            side_state.peclet_over_biot = memb_state.peclet_number / side_state.biot_number
            side_state.alpha = 1 - (1 if self.eod_parallel_to_sorption else 0) / side_state.biot_number / side_state.non_dim_equiv_resistance
  
    def equilibrium_water_content_from_estimated(self, state): 
        for side_state in state.sides: 
            Bi_Rv_star = side_state.non_dim_vapor_resistance * side_state.biot_number
            side_state.eq_water_content = (Bi_Rv_star * side_state.membrane_interface_water_content 
                                           + side_state.estimated_water_content) / (Bi_Rv_star + 1)

    def update_membrane_water_fluxes(self, state):
        """Calculate membrane water fluxes for both sides. Reads/writes state."""
        for side_state in state.sides: 
            side_state.membrane_water_flux = (
                side_state.biot_number * 
                (side_state.membrane_interface_water_content 
                - side_state.eq_water_content) 
                / state.membrane.water_diffusion_resistance
            )
        

    def solve_membrane_water_balance(self, cell, state=None, water_profile=None, dynamic=False):
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
      
        state.membrane.vapor_equilibrium_saturation_water_content = cell.membrane.equilibrium_water_content(
            rh=1., temperature=state.membrane.temperature)
        
        self.calculate_membrane_transport_properties(cell, state)
        self.estimate_equilibrium_water_contents(cell, state)
        self.update_non_dimensional_parameters(cell, state)
        self.update_water_profile(state)
        self.update_water_contents(state)
        self.equilibrium_water_content_from_estimated(state)
        self.update_membrane_water_fluxes(state)
        return self.water_content_profile


    def update_water_profile(self, state):
        """Calculate the water content profile across the membrane. Reads from *state*."""
        Pe   = state.membrane.peclet_number
        ePe  = state.membrane.ePe
        ePexi = state.membrane.ePexi

        alpha_an = state.an.alpha
        alpha_ca = state.ca.alpha
        Pe_over_mod_Bi_an  = state.an.peclet_over_modified_biot
        Pe_over_mod_Bi_ca  = state.ca.peclet_over_modified_biot
        lmbd_est_an   = state.an.estimated_water_content
        lmbd_est_ca   = state.ca.estimated_water_content

        denominator = (
            (ePe - 1) * (
                1 + alpha_an * Pe_over_mod_Bi_an - alpha_ca * Pe_over_mod_Bi_ca
                - alpha_ca * alpha_an * Pe_over_mod_Bi_an * Pe_over_mod_Bi_ca
            )
            + Pe_over_mod_Bi_an + ePe * Pe_over_mod_Bi_ca
            + Pe_over_mod_Bi_an * Pe_over_mod_Bi_ca * (ePe * alpha_an - alpha_ca)
        )

        self.water_content_profile = (
            (
                lmbd_est_an * ((ePe - ePexi) * (1 - alpha_ca * Pe_over_mod_Bi_ca) + ePe * Pe_over_mod_Bi_ca)
                + lmbd_est_ca * ((ePexi - 1) * (1 + alpha_an * Pe_over_mod_Bi_an) + Pe_over_mod_Bi_an)
            ) / denominator
        )
        self.water_content_derivative_profile = (
            (
                lmbd_est_an * (ePexi * Pe * (alpha_ca * Pe_over_mod_Bi_ca - 1))
                + lmbd_est_ca * (ePexi * Pe * (1 + alpha_an * Pe_over_mod_Bi_an))
            ) / denominator
        )
