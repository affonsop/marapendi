"""
Membrane water balance model.

:class:`MembraneWaterBalanceModel` computes the steady-state water content
profile across the membrane and the resulting water fluxes at the
catalyst-layer/membrane interfaces, given a :class:`~marapendi.cell.Cell`
and its :class:`~marapendi.state.CellState`.

All physical variables are read from ``state`` (or computed from
``cell.membrane`` correlations); only the steady-state solution path is
implemented.

Notes
-----
The class is based on the equations and assumptions in Ferrara et al. (2018),
extended to account for gas transport resistance and non-equilibrium
conditions at the membrane interfaces.

References
----------
Ferrara, A. et al. J. Power Sources 390, 197-207 (2018).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cell import Cell
from .gas import GasModel
from .state import CellSideState, CellState
from .transport import GasTransportModel


@dataclass
class MembraneWaterBalanceModel:
    """Steady-state membrane water balance.

    Attributes
    ----------
    sorption_activity_driving_force : bool, optional
        Whether water activity (rather than water content difference) is the
        driving force for water absorption (default is False).
    eod_parallel_to_sorption : bool, optional
        Whether electro-osmotic drag adds to the water absorption flux on
        the right-hand side of the water balance boundary conditions
        (default is False).
    """

    sorption_activity_driving_force: bool = False
    eod_parallel_to_sorption: bool = False

    def calculate_water_transport(self, cell: Cell, state: CellState, gas_transport_model: GasTransportModel) -> None:
        """Update the water vapor transport resistances and solve the membrane water balance.

        For each side, computes the H2O gas transport resistance, solves the
        membrane water balance, and sets the catalyst-layer ionomer water
        content to its equilibrium value. The resulting cathode liquid water
        flux is then used to update the liquid water saturation of the
        cathode porous layers, and the H2O gas transport resistances are
        recomputed to reflect it.
        """
        for side, side_state in zip(cell.sides, state.sides):
            side_state.h2ov_transport_resistance = gas_transport_model.gas_transport_resistance(
                side, side_state, 'h2o',
            )

        self.solve_water_balance(cell, state)

        for side_state in state.sides:
            side_state.cl.ionomer_water_content = side_state.cl.eq_water_content

        self.calculate_water_saturation(cell, state)

        for side, side_state in zip(cell.sides, state.sides):
            side_state.h2ov_transport_resistance = gas_transport_model.gas_transport_resistance(
                side, side_state, 'h2o',
            )

    def calculate_water_saturation(self, cell: Cell, state: CellState) -> None:
        """Update the liquid water saturation of the cathode porous layers.

        Cathode-only, following Ferrara et al. (2018).

        The catalyst layer's non-wetting saturation has already been set by
        :meth:`_solve_liquid_equilibrated_balance`. The GDL's (and MPL's, if
        present) non-wetting saturation is derived here from the cathode
        liquid water flux via :meth:`PorousLayer.non_wetting_saturation_from_flux`,
        chaining the downstream capillary pressure from GDL to MPL. The
        liquid saturation of every cathode porous layer is then set from its
        non-wetting saturation, according to its wettability.
        """
        ca, ca_state = cell.ca, state.ca

        upstream_capillary_pressure = 0.
        for layer, layer_state in zip((ca.gdl, ca.mpl) if ca.has_mpl else (ca.gdl,), (ca_state.gdl, ca_state.mpl) if ca.has_mpl else (ca_state.gdl,)):
            layer_state.non_wetting_saturation, upstream_capillary_pressure = layer.non_wetting_saturation_from_flux(
                layer_state, ca_state.liquid_flux, upstream_capillary_pressure,
            )

        for layer, layer_state in zip(ca.porous_layers, ca_state.porous_layers[::-1]):
            layer_state.liquid_saturation = (
                layer_state.non_wetting_saturation if layer.contact_angle > 90.
                else 1. - layer_state.non_wetting_saturation
            )

    def calculate_peclet_number(self, membrane, temperature: float, current_density: float,
                                 water_diffusivity: float) -> float:
        """Peclet number from the electroosmotic drag speed, membrane thickness and water diffusivity."""
        return (
            membrane.calculate_electroosmotic_drag_speed(temperature, current_density)
            * membrane.dry_thickness / water_diffusivity
        )

    def calculate_biot_number(self, absorption_coefficient: float, membrane, water_diffusivity: float) -> float:
        """Biot number from the absorption coefficient, membrane thickness and water diffusivity."""
        return absorption_coefficient * membrane.dry_thickness / water_diffusivity

    def solve_water_balance(self, cell: Cell, state: CellState):
        """Solve the steady-state membrane water balance and update ``state`` in place.

        Returns
        -------
        np.ndarray
            The membrane water content profile (also stored in
            ``state.membrane.water_content_profile``).
        """
        membrane, membrane_state = cell.membrane, state.membrane

        self._initialize_saturation_water_contents(cell, state)
        self._initialize_interface_water_contents(state)

        self.absorption_coefficient = membrane.calculate_water_absorption_coefficient(membrane_state.temperature)
        self.water_diffusivity = membrane.calculate_water_diffusivity(membrane_state.temperature)
        membrane_state.water_diffusion_resistance = (
            membrane.dry_thickness / (self.water_diffusivity * membrane.dry_concentration)
        )

        self.estimate_equilibrium_water_contents(cell, state)
        self.update_non_dimensional_parameters(cell, state)
        self._solve_static_water_balance(cell, state)

        return membrane_state.water_content_profile

    def estimate_equilibrium_water_contents(self, cell: Cell, state: CellState) -> None:
        """Estimate the catalyst-layer equilibrium water content and its derivative,
        assuming no water crossover from cathode to anode."""
        membrane, membrane_state = cell.membrane, state.membrane

        for side_state in state.sides:
            side_state.rh_at_cl_without_crossover = (
                (GasModel.vapor_concentration(side_state.ch)
                 + side_state.h2o_production * side_state.h2ov_transport_resistance)
                / GasModel.saturation_concentration(side_state.cl)
            )
            side_state.estimated_water_content = membrane.equilibrium_water_content(
                side_state.rh_at_cl_without_crossover, membrane_state.temperature, side_state.s_relax,
            )
            side_state.estimated_water_content_derivative = membrane.equilibrium_water_content_derivative(
                side_state.rh_at_cl_without_crossover, membrane_state.temperature, side_state.s_relax,
            )

    def update_non_dimensional_parameters(self, cell: Cell, state: CellState) -> None:
        """Compute the Peclet number, Biot number and other non-dimensional parameters
        used by :meth:`update_water_profile`."""
        membrane, membrane_state = cell.membrane, state.membrane

        membrane_state.peclet_number = self.calculate_peclet_number(
            membrane, membrane_state.temperature, state.current_density, self.water_diffusivity,
        )
        membrane_state.xi = np.linspace(0, np.ones_like(state.current_density), 10)
        membrane_state.ePexi = np.exp(membrane_state.xi * membrane_state.peclet_number)
        membrane_state.ePe = np.exp(membrane_state.peclet_number)

        for side_state in state.sides:
            side_state.R_v_star = (
                side_state.h2ov_transport_resistance
                / (GasModel.saturation_concentration(side_state.cl) * membrane_state.water_diffusion_resistance)
                * side_state.estimated_water_content_derivative
            )
            side_state.biot_number = self.calculate_biot_number(
                self.absorption_coefficient
                / (side_state.estimated_water_content_derivative if self.sorption_activity_driving_force else 1),
                membrane, self.water_diffusivity,
            )
            side_state.R_eq_star = side_state.R_v_star + 1 / side_state.biot_number
            side_state.modified_Bi = 1 / side_state.R_eq_star
            side_state.peclet_over_modified_biot = membrane_state.peclet_number / side_state.modified_Bi
            side_state.alpha = (
                1 - (1 if self.eod_parallel_to_sorption else 0) / side_state.biot_number / side_state.R_eq_star
            )

    def update_water_profile(self, cell: Cell, state: CellState) -> None:
        """Compute the membrane water content profile and its derivative
        from the cathode/anode equilibrium water contents and non-dimensional parameters."""
        membrane_state = state.membrane
        ca, an = state.ca, state.an

        Pe = membrane_state.peclet_number
        ePe = membrane_state.ePe
        ePexi = membrane_state.ePexi

        denominator = (
            (ePe - 1) * (
                1 + an.alpha * an.peclet_over_modified_biot - ca.alpha * ca.peclet_over_modified_biot
                - ca.alpha * an.alpha * an.peclet_over_modified_biot * ca.peclet_over_modified_biot
            )
            + an.peclet_over_modified_biot + ePe * ca.peclet_over_modified_biot
            + an.peclet_over_modified_biot * ca.peclet_over_modified_biot * (ePe * an.alpha - ca.alpha)
        )

        membrane_state.water_content_profile = (
            (
                an.estimated_water_content
                * ((ePe - ePexi) * (1 - ca.alpha * ca.peclet_over_modified_biot) + ePe * ca.peclet_over_modified_biot)
                + ca.estimated_water_content
                * ((ePexi - 1) * (1 + an.alpha * an.peclet_over_modified_biot) + an.peclet_over_modified_biot)
            ) / denominator
        )
        membrane_state.water_content_derivative_profile = (
            (
                an.estimated_water_content * (ePexi * Pe * (ca.alpha * ca.peclet_over_modified_biot - 1))
                + ca.estimated_water_content * (ePexi * Pe * (1 + an.alpha * an.peclet_over_modified_biot))
            ) / denominator
        )

    def calculate_cathode_membrane_flux(self, state: CellState) -> float:
        """Water flux at the cathode/membrane interface (kmol/m^2/s)."""
        membrane_state = state.membrane
        non_dimensional_membrane_water_flux = (
            -membrane_state.water_content_derivative_profile[-1, ...]
            + membrane_state.peclet_number * membrane_state.water_content_profile[-1, ...]
        )
        return non_dimensional_membrane_water_flux / membrane_state.water_diffusion_resistance

    def calculate_anode_membrane_flux(self, state: CellState) -> float:
        """Water flux at the anode/membrane interface (kmol/m^2/s)."""
        membrane_state = state.membrane
        non_dimensional_membrane_water_flux = (
            membrane_state.water_content_derivative_profile[0, ...]
            - membrane_state.peclet_number * membrane_state.water_content_profile[0, ...]
        )
        return non_dimensional_membrane_water_flux / membrane_state.water_diffusion_resistance

    def update_cell_side_water_fluxes(self, side_state: CellSideState) -> None:
        """Split the total water flux of ``side_state`` into liquid and vapor fluxes,
        limited by the maximum vapor removal rate of the flow channel."""
        max_vapor_removal_flux = (
            (GasModel.saturation_concentration(side_state.cl) - GasModel.vapor_concentration(side_state.ch))
            / side_state.h2ov_transport_resistance
        )
        side_state.liquid_flux = np.maximum(side_state.water_flux - max_vapor_removal_flux, 0)
        side_state.vapor_flux = side_state.water_flux - side_state.liquid_flux

    def update_water_fluxes(self, cell: Cell, state: CellState) -> None:
        """Compute the membrane, total, liquid and vapor water fluxes for both sides."""
        state.ca.membrane_water_flux = self.calculate_cathode_membrane_flux(state)
        state.an.membrane_water_flux = self.calculate_anode_membrane_flux(state)
        state.ca.water_flux = state.ca.h2o_production + state.ca.membrane_water_flux
        state.an.water_flux = state.an.h2o_production + state.an.membrane_water_flux

        self.update_cell_side_water_fluxes(state.ca)
        self.update_cell_side_water_fluxes(state.an)

    def update_water_contents(self, cell: Cell, state: CellState) -> None:
        """Update the membrane average water content and the catalyst-layer
        equilibrium water contents from the current water content profile."""
        membrane, membrane_state = cell.membrane, state.membrane

        membrane_state.water_content = np.mean(membrane_state.water_content_profile, axis=0)
        state.ca.cl.membrane_interface_water_content = membrane_state.water_content_profile[-1, ...]
        state.an.cl.membrane_interface_water_content = membrane_state.water_content_profile[0, ...]

        for side_state in state.sides:
            side_state.cl.eq_water_content = (
                side_state.cl.membrane_interface_water_content
                - side_state.membrane_water_flux / self.absorption_coefficient / membrane.dry_concentration
            )

    def _initialize_saturation_water_contents(self, cell: Cell, state: CellState) -> None:
        """Cache vapor- and liquid-equilibrium water contents at full saturation for use as boundary limits."""
        membrane, membrane_state = cell.membrane, state.membrane
        self.vapor_equilibrium_saturation_water_content = membrane.equilibrium_water_content(
            rh=1., temperature=membrane_state.temperature,
        )
        self.liquid_equilibrium_saturation_water_content = membrane.liquid_equilibrium_water_content(
            temperature=membrane_state.temperature,
        )

    def _initialize_interface_water_contents(self, state: CellState) -> None:
        """Set the membrane/catalyst-layer interface water contents to zero (steady-state initial guess)."""
        for side_state in state.sides:
            side_state.cl.membrane_interface_water_content = 0

    def _solve_static_water_balance(self, cell: Cell, state: CellState) -> None:
        """Solve the steady-state water balance, branching into liquid-equilibrated treatment if needed."""
        self.update_water_profile(cell, state)
        self.update_water_fluxes(cell, state)
        self.update_water_contents(cell, state)

        state.ca.is_liquid_equilibrated = state.ca.liquid_flux > 0

        if np.any(state.ca.is_liquid_equilibrated):
            self._solve_liquid_equilibrated_balance(cell, state)
        else:
            self._store_vapor_equilibrium_state(state)
            self._store_liquid_equilibrium_state(state)

    def _solve_liquid_equilibrated_balance(self, cell: Cell, state: CellState) -> None:
        """Three-stage solution for liquid-equilibrated cathode locations:

        1. Apply the vapor-equilibrium boundary condition and store the resulting state.
        2. Apply the liquid-equilibrium boundary condition and store the resulting state.
        3. Blend the two states, weighted by non-wetting saturation.
        """
        self._store_vapor_equilibrium_state(state)

        self._apply_liquid_equilibrium_condition(cell, state)
        self._store_liquid_equilibrium_state(state)

        state.ca.cl.non_wetting_saturation[state.ca.is_liquid_equilibrated] = (
            self._compute_non_wetting_saturation(cell, state)
        )

        self._blend_equilibrium_profiles(state)
        self._update_blended_fluxes(state)

        self.update_cell_side_water_fluxes(state.ca)
        self.update_cell_side_water_fluxes(state.an)

    def _store_vapor_equilibrium_state(self, state: CellState) -> None:
        """Store water fluxes and profiles after applying the vapor-equilibrium condition."""
        ca, membrane_state = state.ca, state.membrane
        ca.vapor_eq_sat_membrane_water_flux = ca.membrane_water_flux
        ca.vapor_eq_sat_water_flux = ca.water_flux
        ca.vapor_eq_sat_liquid_flux = ca.liquid_flux
        ca.vapor_eq_water_content = ca.cl.eq_water_content
        membrane_state.vapor_eq_sat_water_profile = membrane_state.water_content_profile
        membrane_state.vapor_eq_sat_water_derivative_profile = membrane_state.water_content_derivative_profile

    def _apply_liquid_equilibrium_condition(self, cell: Cell, state: CellState) -> None:
        """Impose the liquid-equilibrium water content at liquid-equilibrated cathode locations
        and recompute the water profile and fluxes.

        The Peclet/Biot ratio is divided by 100 to represent the much higher
        water activity at the membrane interface under liquid equilibrium.
        """
        ca = state.ca
        liquid_equilibrated_mask = ca.is_liquid_equilibrated
        ca.peclet_over_modified_biot = np.where(
            liquid_equilibrated_mask,
            state.membrane.peclet_number / ca.biot_number / 100,
            ca.peclet_over_modified_biot,
        )
        ca.estimated_water_content = np.where(
            liquid_equilibrated_mask,
            self.liquid_equilibrium_saturation_water_content,
            ca.estimated_water_content,
        )
        self.update_water_profile(cell, state)
        self.update_water_fluxes(cell, state)
        self.update_water_contents(cell, state)

    def _store_liquid_equilibrium_state(self, state: CellState) -> None:
        """Store water fluxes and profiles after applying the liquid-equilibrium condition."""
        ca, membrane_state = state.ca, state.membrane
        ca.liquid_eq_sat_membrane_water_flux = ca.membrane_water_flux
        ca.liquid_eq_sat_water_flux = ca.water_flux
        ca.liquid_eq_sat_liquid_flux = ca.liquid_flux
        ca.liquid_eq_water_content = ca.cl.eq_water_content
        membrane_state.liquid_eq_sat_water_profile = membrane_state.water_content_profile
        membrane_state.liquid_eq_sat_water_derivative_profile = membrane_state.water_content_derivative_profile

    def _compute_non_wetting_saturation(self, cell: Cell, state: CellState, n_iter: int = 5):
        """Estimate the non-wetting saturation at liquid-equilibrated cathode locations."""
        liquid_equilibrated_mask = state.ca.is_liquid_equilibrated
        modified_abs_permeability, membrane_flux_difference, vapor_eq_liquid_flux, combined_flux_exponent = (
            self._assemble_saturation_equation_params(cell, state, liquid_equilibrated_mask)
        )
        return self._halley_solve_saturation(
            modified_abs_permeability,
            membrane_flux_difference,
            vapor_eq_liquid_flux,
            combined_flux_exponent,
            n_iter,
        )

    def _assemble_saturation_equation_params(self, cell: Cell, state: CellState, liquid_equilibrated_mask):
        """Assemble the coefficients of the non-wetting saturation equation:

            modified_abs_permeability * s^combined_flux_exponent + membrane_flux_difference * s
                - vapor_eq_liquid_flux = 0

        Returns
        -------
        modified_abs_permeability : np.ndarray
            Effective absolute permeability combining GDL and CL transport properties.
        membrane_flux_difference : np.ndarray
            Difference in membrane water flux between vapor- and liquid-equilibrium states.
        vapor_eq_liquid_flux : np.ndarray
            Liquid water flux under vapor-equilibrium conditions.
        combined_flux_exponent : float
            Combined exponent m+n from the GDL relative permeability and CL/GDL J-function exponents.
        """
        cathode, cathode_state = cell.ca, state.ca

        gdl_permeability_exponent = cathode.gdl.relative_permeability_exponent
        cl_J_function_exponent = cathode.cl.two_phase_transport_model.J_function_exponent
        gdl_J_function_exponent = cathode.gdl.two_phase_transport_model.J_function_exponent

        combined_flux_exponent = (
            gdl_permeability_exponent * cl_J_function_exponent / gdl_J_function_exponent
            + cl_J_function_exponent
        )

        cl_to_gdl_capillary_pressure_ratio = (
            cathode.cl.capillary_pressure_J_ratio(cathode_state.cl)
            / cathode.gdl.capillary_pressure_J_ratio(cathode_state.gdl)
        )
        modified_abs_permeability = (
            1. / cathode.gdl.saturation_flow_resistance(cathode_state.gdl)
            * gdl_J_function_exponent / (gdl_permeability_exponent + gdl_J_function_exponent)
            * cl_to_gdl_capillary_pressure_ratio ** (gdl_permeability_exponent / gdl_J_function_exponent + 1)
        )[liquid_equilibrated_mask]

        membrane_flux_difference = (
            cathode_state.vapor_eq_sat_membrane_water_flux
            - cathode_state.liquid_eq_sat_membrane_water_flux
        )[liquid_equilibrated_mask]

        vapor_eq_liquid_flux = cathode_state.vapor_eq_sat_liquid_flux[liquid_equilibrated_mask]

        return modified_abs_permeability, membrane_flux_difference, vapor_eq_liquid_flux, combined_flux_exponent

    @staticmethod
    def _halley_solve_saturation(modified_abs_permeability, membrane_flux_difference,
                                  vapor_eq_liquid_flux, combined_flux_exponent, n_iter=5):
        """Halley's method root-finder for the non-wetting saturation equation:

            f(s) = modified_abs_permeability * s^combined_flux_exponent
                + membrane_flux_difference * s - vapor_eq_liquid_flux = 0

        Initial guess of 0.1 is reasonable for low saturations. Saturation is
        clipped to [0, 0.9] each iteration to remain in the physical range.
        """
        saturation = np.full_like(membrane_flux_difference, 0.1)
        for _ in range(n_iter):
            residual = (
                modified_abs_permeability * saturation ** combined_flux_exponent
                + membrane_flux_difference * saturation
                - vapor_eq_liquid_flux
            )
            first_deriv = (
                combined_flux_exponent * modified_abs_permeability * saturation ** (combined_flux_exponent - 1)
                + membrane_flux_difference
            )
            second_deriv = (
                combined_flux_exponent * (combined_flux_exponent - 1)
                * modified_abs_permeability * saturation ** (combined_flux_exponent - 2)
            )
            saturation = np.clip(
                saturation - residual * first_deriv / (first_deriv ** 2 - 0.5 * residual * second_deriv),
                0, 0.9,
            )
        return saturation

    def _blend_equilibrium_profiles(self, state: CellState) -> None:
        """Linearly interpolate water content profiles between vapor- and liquid-equilibrium states,
        weighted by non-wetting saturation."""
        ca, membrane_state = state.ca, state.membrane
        non_wetting_saturation = ca.cl.non_wetting_saturation

        membrane_state.water_content_profile = (
            membrane_state.vapor_eq_sat_water_profile * (1 - non_wetting_saturation)
            + membrane_state.liquid_eq_sat_water_profile * non_wetting_saturation
        )
        membrane_state.water_content_derivative_profile = (
            membrane_state.vapor_eq_sat_water_derivative_profile * (1 - non_wetting_saturation)
            + membrane_state.liquid_eq_sat_water_derivative_profile * non_wetting_saturation
        )
        ca.cl.eq_water_content = (
            ca.vapor_eq_water_content * (1 - non_wetting_saturation)
            + ca.liquid_eq_water_content * non_wetting_saturation
        )
        membrane_state.water_content = np.mean(membrane_state.water_content_profile, axis=0)

    def _update_blended_fluxes(self, state: CellState) -> None:
        """Update the membrane and total water fluxes using the blended non-wetting saturation."""
        ca, an = state.ca, state.an
        membrane_flux_difference = ca.vapor_eq_sat_membrane_water_flux - ca.liquid_eq_sat_membrane_water_flux
        non_wetting_saturation = ca.cl.non_wetting_saturation

        ca.membrane_water_flux = ca.vapor_eq_sat_membrane_water_flux - membrane_flux_difference * non_wetting_saturation
        an.membrane_water_flux = -ca.membrane_water_flux
        ca.water_flux = ca.h2o_production + ca.membrane_water_flux
        an.water_flux = an.h2o_production + an.membrane_water_flux
