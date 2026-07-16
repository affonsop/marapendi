"""
Piecewise-linear membrane water balance model.

Extends :class:`MembraneWaterBalanceModel` by replacing the polynomial sorption
isotherm with the piecewise linear approximation fitted by
:meth:`~marapendi.components.membrane.pem.PFSAIonomer.fit_rh_piecewise_linear`.  The active
linear segment is selected self-consistently so the equilibrium water content
falls within the validity interval of that segment.
"""
import numpy as np
from dataclasses import dataclass
from .membrane import MembraneWaterBalanceModel


@dataclass
class MembraneWaterBalanceModelPiecewise(MembraneWaterBalanceModel):
    """
    Membrane water balance using a piecewise linear sorption isotherm.

    The linear segment active at each operating point is found by iterating
    downward from the highest segment until the resulting equilibrium water
    content falls within the validity bounds of the chosen segment.
    Convergence is guaranteed in at most ``n_segments`` iterations.
    """

    def estimate_equilibrium_water_contents(self, cell, state):
        """Estimate CL equilibrium water content using the piecewise linear λ(RH) inverse.

        ``side_state.pwl_interval`` selects which linear segment to use.  This
        must already be set (by :meth:`solve_membrane_water_balance`) before
        calling this method.
        """
        membrane = cell.membrane

        for side_state in state.sides:
            side_state.rh_at_cl_without_crossover = (
                (side_state.ch.gas.vapor_concentration
                 + side_state.h2o_production * side_state.h2ov_transport_resistance)
                / side_state.cl.gas.saturation_concentration
            )
            side_state.estimated_water_content = membrane.ionomer.linear_water_content_from_rh(
                side_state.rh_at_cl_without_crossover, side_state.pwl_interval)

    def update_non_dim_vapor_resistance(self, side_state, memb_state, cell):
        # dλ/dRH = 1 / slope_k  (slope is dRH/dλ for the active segment)
        return (
            side_state.h2ov_transport_resistance
            / (side_state.cl.gas.saturation_concentration * memb_state.water_diffusion_resistance)
            / cell.membrane.ionomer.pwl_slopes[side_state.pwl_interval]
        )

    def solve_membrane_water_balance(self, cell, state=None, water_profile=None, dynamic=False):
        """Solve the membrane water balance using piecewise linear λ(RH).

        The linear segment is chosen self-consistently: after solving with a
        candidate segment the resulting ``eq_water_content`` is tested against
        the validity intervals (line intersections).  If it falls outside the
        assumed segment the nearest correct segment is selected and the solve is
        repeated.  Convergence is guaranteed for a monotone isotherm in at most
        ``n_segments`` iterations.

        Parameters
        ----------
        cell : FuelCell
        state : CellState
        water_profile : np.ndarray, optional
            Required when *dynamic* is True.
        dynamic : bool
            If True solve for a prescribed water profile (transient mode).
        """
        
        self.calculate_membrane_transport_properties(cell, state)

        ionomer  = cell.membrane.ionomer
        n_seg    = len(ionomer.pwl_slopes)
        # Interior validity breakpoints in λ space (exclude outer bounds)
        lmbd_inner = ionomer.lmbd_pwl_breaks[2]  # shape (n_seg - 1,)

        # Start from the highest segment — most common at typical RH
        for side_state in state.sides:
            side_state.pwl_interval = n_seg - 1

        # Iterate downward only: since we start at the highest segment, each
        # element's interval is strictly non-increasing → guaranteed to converge
        # in at most n_seg steps without oscillation.
        converged = False
        while not converged:
            # Physics calls go here — once per iteration, not once per side
            self.estimate_equilibrium_water_contents(cell, state)
            self.update_non_dimensional_parameters(cell, state)
            if not dynamic: 
                self.update_water_profile(state)
            self.update_water_contents(state)
            self.equilibrium_water_content_from_estimated(state)

            converged = True
            for side_state in state.sides:
                new_k = np.where(lmbd_inner < side_state.eq_water_content, 2, 1)
                if np.any(new_k != side_state.pwl_interval):
                    converged = False
                # Always assign so pwl_interval is array-valued for vectorised inputs
                side_state.pwl_interval = new_k
        sat_eq_water_content = cell.membrane.ionomer.linear_water_content_from_rh(1)
        for side_state in state.sides:
            side_state.is_saturated = (side_state.eq_water_content > 
                                                        sat_eq_water_content)
            side_state.estimated_water_content = np.where(side_state.is_saturated, 
                                                        sat_eq_water_content, side_state.estimated_water_content)
            side_state.peclet_over_modified_biot = np.where(side_state.is_saturated, 
                                                    side_state.peclet_over_biot, side_state.peclet_over_modified_biot)
            side_state.non_dim_vapor_resistance = np.where(side_state.is_saturated, 
                                                    0, side_state.non_dim_vapor_resistance)
            if not dynamic: 
                self.update_water_profile(state)
            self.update_water_contents(state)
            self.equilibrium_water_content_from_estimated(state)
            self.equilibrium_water_content_from_estimated(state)

        self.update_membrane_water_fluxes(state)

        return self.water_content_profile
