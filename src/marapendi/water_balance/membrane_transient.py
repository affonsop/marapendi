
import numpy as np
from dataclasses import dataclass, field
from marapendi.thermo.constants import GAS_CONSTANT
from marapendi.thermo.gas import GasModel
from marapendi.tools import arrhenius_term
from marapendi.thermo.water import water_molar_volume, water_dynamic_viscosity
from ..cell.gas_transport import GasTransportModel
from .membrane import MembraneWaterBalanceModel 

@dataclass
class MembraneWaterBalanceTransientModel(MembraneWaterBalanceModel):

    def _initialize_interface_water_contents(self, state, water_profile):
        """Set membrane interface water contents from profile (dynamic) or zero (static)."""
        
        self.water_content_profile = water_profile
        state.ca.membrane_interface_water_content = water_profile[-1, ...]
        state.an.membrane_interface_water_content = water_profile[0, ...]

    def update_internal_water_fluxes(self, state, cell):
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


    def solve_membrane_water_balance(self, cell, state=None, water_profile=None, dynamic=False):
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

        state.membrane.vapor_equilibrium_saturation_water_content = (
            cell.membrane.equilibrium_water_content(rh=1., temperature=state.membrane.temperature)
        )

        self._initialize_interface_water_contents(state, water_profile)
        self.calculate_membrane_transport_properties(cell, state)
        self.estimate_equilibrium_water_contents(cell, state)
        self.update_non_dimensional_parameters(cell, state)
        self.update_water_contents(state)
        self.equilibrium_water_content_from_estimated(state)
        self.update_membrane_water_fluxes(state)

        return self.water_content_profile