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
from dataclasses import dataclass, field
from ..thermo.constants import GAS_CONSTANT

from ...tools import arrhenius_term
from ..thermo.water import water_molar_volume, water_dynamic_viscosity
from ..gas_transport_resistance import GasTransportModel
from .membrane import * 
from .membrane_pwl import *

@dataclass
class WaterBalanceModel:
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
    Based on Ferrara et al. (2018), accounting for gas transport resistance
    and non-equilibrium sorption at the membrane interface.

    References
    ----------
    Ferrara, A. et al. J. Power Sources 390, 197–207 (2018).
    """
    
    sorption_activity_driving_force: bool = False
    eod_parallel_to_sorption: bool = False
    membrane_water_balance_model: MembraneWaterBalanceModel = field(default_factory=MembraneWaterBalanceModelPiecewise)

    def update_cell_side_water_fluxes(self, side_state) -> None:
        """Split total water flux into liquid and vapor components.

        Liquid flux = excess above the maximum vapor removal capacity of the
        channel (at the current saturation state of the CL).
        """
        side_state.max_vapor_removal_flux = (
            (side_state.cl.gas.saturation_concentration() - side_state.ch.gas.vapor_concentration())
            / side_state.h2ov_transport_resistance
        )

        side_state.water_flux = side_state.membrane_water_flux + side_state.h2o_production
        side_state.liquid_flux = np.maximum(side_state.water_flux - side_state.max_vapor_removal_flux, 0)
        side_state.vapor_flux  = side_state.water_flux - side_state.liquid_flux
        side_state.gas_flux    = side_state.vapor_flux

    
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

        for layer, ls in zip(cell_side.porous_layers, side_state.porous_layers):
            ls.liquid_saturation = (
                ls.non_wetting_saturation if layer.contact_angle > 90.
                else (1 - ls.non_wetting_saturation)
            )
            ls.electrolyte_saturation = ls.liquid_saturation


    def calculate_water_transport(self, cell, state, dynamic: bool = False,
                                   water_profile=None, gas_transport_model=None) -> None:
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
            models that prescribe the water profile externally).
        water_profile : np.ndarray, optional
            Prescribed membrane water-content profile (shape ``(n_mesh, ...)``).
            Passed through to :meth:`MembraneWaterBalanceModel.solve_membrane_water_balance`;
            when ``None`` the steady-state profile is computed internally.
        gas_transport_model : GasTransportModel, optional
            Shared instance for computing H₂O vapor transport resistances.
            A temporary instance is created if not provided.
        """
        _gtr = gas_transport_model if gas_transport_model is not None else GasTransportModel()
        htr_ca = _gtr.gas_transport_resistance(cell.ca, state.ca, 'h2o')
        htr_an = _gtr.gas_transport_resistance(cell.an, state.an, 'h2o')
        state.ca.h2ov_transport_resistance = htr_ca
        state.an.h2ov_transport_resistance = htr_an

        self.membrane_water_balance_model.solve_membrane_water_balance(
            cell, state, water_profile=water_profile
        )
        self.update_cell_side_water_fluxes(state.ca)
        self.update_cell_side_water_fluxes(state.an)

        if not dynamic:
            self.calculate_water_saturation(cell.ca, state.ca)
            cell.ca.cl.set_water_film_thickness(state.ca.cl.non_wetting_saturation)
            htr_ca = _gtr.gas_transport_resistance(cell.ca, state.ca, 'h2o')
            htr_an = _gtr.gas_transport_resistance(cell.an, state.an, 'h2o')
            state.ca.h2ov_transport_resistance = htr_ca
            state.an.h2ov_transport_resistance = htr_an

        for cl, side_state in [(cell.ca.cl, state.ca), (cell.an.cl, state.an)]:
            if cell.use_eq_water_content_for_ionomer:
                side_state.cl.ionomer_water_content = side_state.eq_water_content
            else:
                side_state.cl.ionomer_water_content = side_state.membrane_interface_water_content
            cl.set_ionomer_wet_properties(side_state.cl.ionomer_water_content, side_state.cl.temperature)

    def membrane_water_rate_of_change(self, cell, state, n_memb_mesh):
        """Rate of change of water content at each membrane mesh node (1/s).

        Uses a finite-volume discretisation of the combined diffusion + EOD
        flux divergence, with boundary fluxes taken from the absorption/desorption
        terms already stored in *state* by :meth:`calculate_water_transport`.

        Parameters
        ----------
        cell : FuelCell
        state : CellState
            Must have ``membrane.water_content_profile``, ``membrane.peclet_number``,
            ``membrane.water_diffusion_resistance``, ``an.membrane_water_flux``, and
            ``ca.membrane_water_flux`` already populated.
        n_memb_mesh : int
            Number of membrane mesh nodes (must match ``len(state.membrane.water_content_profile)``).

        Returns
        -------
        np.ndarray, shape (n_memb_mesh, ...)
        """
        lmbd = state.membrane.water_content_profile   # (n, ...)
        R_D = state.membrane.water_diffusion_resistance
        Pe = state.membrane.peclet_number
        n = n_memb_mesh
        sc = cell.membrane.surface_concentration       # kmol/m²

        # Interior face fluxes J(k+1/2), positive toward cathode (anode→cathode), shape (n-1, ...)
        J_int = (
            -n * np.diff(lmbd, axis=0) + Pe * (lmbd[:-1] + lmbd[1:]) / 2
        ) / R_D

        # Boundary face fluxes (positive = in anode→cathode direction)
        # J_{-1/2}: entering membrane from anode = –(flux leaving to anode)
        # J_{n-1/2}: leaving membrane to cathode = +(flux leaving to cathode)
        # Reshape boundary fluxes to match J_int's trailing shape (n_current, …)
        trailing = lmbd.shape[1:]
        J_left = np.reshape(-state.an.membrane_water_flux, (1,) + trailing)
        J_right = np.reshape(state.ca.membrane_water_flux, (1,) + trailing)

        J_faces = np.concatenate([J_left, J_int, J_right], axis=0)

        # Accumulation: J_in – J_out at each node
        return (J_faces[:-1] - J_faces[1:]) / (sc / n)

    def relaxation_rate_of_change(self, cell, state):
        """Rate of change of the ionomer sorption relaxation variable on each side (1/s).

        Returns a 2-element list ``[dsrelax_ca_dt, dsrelax_an_dt]``.
        Requires ``state.*.s_relax`` and ``state.*.rh_at_cl_without_crossover`` to be set.
        """
        from ...tools import arrhenius_term
        from ..thermo.constants import GAS_CONSTANT

        result = []
        membrane = cell.membrane
        phi = getattr(membrane, 'phi', 0.)
        tau0 = getattr(membrane, 'relaxation_time_constant', 1.)
        Ea = getattr(membrane, 'relaxation_time_activation_energy', 0.)
        T = state.membrane.temperature

        for side_state in state.sides:
            if side_state.s_relax is None:
                result.append(np.zeros_like(state.current_density))
                continue
            lmbd_springer = membrane.ionomer.vapor_equilibrium_water_content(
                side_state.rh_at_cl_without_crossover, T
            )
            s_target = phi * lmbd_springer
            tau = tau0 * np.exp(Ea / (GAS_CONSTANT * T))
            result.append((s_target - side_state.s_relax) / tau)

        return result
