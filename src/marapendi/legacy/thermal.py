from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cantera as ct

from .electrochemistry import calculate_reversible_cell_voltage, STD_PRESSURE
from .state import FuelCellState
from .water import water_density, water_molecular_weight, water_surface_tension
from ..legacy.water import water_saturation_concentration

@dataclass
class ThermalModel:

    def set_mea_temperature(self, mea_temperature, state): 
        """
        Set the membrane electrode assembly (MEA) temperature and update associated components.
        The underlying hypotheses is that the membrane and the CLs are at the
        same temperature.

        Parameters
        ----------
        mea_temperature : float
            The new temperature of the MEA in Kelvin.

        Notes
        -----
        This method updates the temperature of various fuel cell components, including:
        - Cathode catalyst layer (CL)
        - Anode catalyst layer (CL)
        - Membrane

        It also calculates the temperature increase of the MEA relative to the initial temperature.
        """

        mea_csat = water_saturation_concentration(mea_temperature)
        rho_l = water_density(mea_temperature)
        sigma_l = water_surface_tension(mea_temperature)

        for layer in (state.membrane, state.ca.cl, state.an.cl): 
            layer.temperature = mea_temperature
            layer.saturation_concentration = mea_csat 
            layer.water_density = rho_l
            layer.water_molar_volume = water_molecular_weight / rho_l
            layer.water_surface_tension = sigma_l 

