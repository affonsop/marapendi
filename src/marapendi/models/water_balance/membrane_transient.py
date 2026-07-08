"""
Transient membrane water balance model.

Extends :class:`MembraneWaterBalanceModel` to handle a prescribed membrane
water-content profile supplied by the transient ODE integrator.  When the
profile is available, the steady-state analytical solution is bypassed and
boundary fluxes are computed directly from the prescribed profile.
"""
import numpy as np
from dataclasses import dataclass
from .membrane_pwl import MembraneWaterBalanceModelPiecewise


@dataclass
class MembraneWaterBalanceTransientModel(MembraneWaterBalanceModelPiecewise):
    """
    Membrane water balance for transient integration with a prescribed profile.

    When a water-content profile is provided (transient mode), the boundary
    fluxes are evaluated from the prescribed profile rather than solving the
    steady-state analytical equation.  Falls back to the parent steady-state
    solver when no profile is supplied.
    """

    def _initialize_interface_water_contents(self, state, water_profile):
        """Set membrane interface water contents from profile (dynamic) or zero (static)."""
        
        self.water_content_profile = water_profile
        state.ca.membrane_interface_water_content = water_profile[-1, ...]
        state.an.membrane_interface_water_content = water_profile[0, ...]

    def update_internal_water_fluxes(self, state, cell):
        """Compute diffusion, EOD, and net water fluxes across the membrane mesh; write to ``state.membrane``."""
        state.membrane.diffusion_flux = (
            -self.water_content_derivative_profile / state.membrane.water_diffusion_resistance
        )
        state.membrane.eod_flux = state.membrane.peclet_number * self.water_content_profile
        state.membrane.water_flux = state.membrane.diffusion_flux + state.membrane.eod_flux

        state.membrane.water_net_flux = -np.diff(
            state.membrane.diffusion_flux, prepend=0, append=0, axis=0
        )
        state.membrane.water_net_flux[0, ...] -= state.an.membrane_water_flux
        state.membrane.water_net_flux[-1, ...] -= state.ca.membrane_water_flux


    def solve_membrane_water_balance(self, cell, state=None, water_profile=None):
        """Solve the membrane water balance for a prescribed water-content profile.

        When *water_profile* is ``None``, delegates to the steady-state solver in
        the parent class (:meth:`MembraneWaterBalanceModel.solve_membrane_water_balance`).

        Parameters
        ----------
        cell : FuelCell
        state : CellState
        water_profile : np.ndarray, shape (n_mesh, ...), optional
            Prescribed membrane water-content profile.  Must be provided for
            transient integration; if omitted the steady-state profile is solved.
        dynamic : bool
            Unused — retained for backward compatibility.
        """
        if water_profile is None:
            return super().solve_membrane_water_balance(cell, state)
        else: 
            self.water_content_profile = water_profile
        super().solve_membrane_water_balance(cell, state, dynamic=True)
        return self.water_content_profile
    
 