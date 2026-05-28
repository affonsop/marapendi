"""
Module providing classes to model ionomers in catalyst layers.

Classes
-------
CatalystLayerIonomer : Base class for ionomer properties in catalyst layers.
PFSAIonomer : Represents a perfluorosulfonic acid ionomer (e.g. Nafion).
PAPIonomer : Represents a poly(aryl piperidinium) (PA) ionomer.

Each class provides methods to compute water content-dependent properties
like proton or hydroxide conductivity, oxygen permeability, and equilibrium hydration.
"""

from dataclasses import dataclass, field
import numpy as np
import cantera as ct

from marapendi.tools.tools import arrhenius_term, Updatable
from .water import water_molecular_weight, water_molar_volume, water_density


@dataclass 
class Ionomer(Updatable): 
    dry_density: float 
    equivalent_weight: float 
    darken_num: np.ndarray
    darken_den: np.ndarray
    sorption_isotherm_coeffs: np.ndarray
    reference_liquid_equilibrium_water_content: float 
    reference_water_diffusivity: float
    reference_desorption_coefficient: float 
    ionomer_activation_energy: float
    reference_charge_conductivity: float = 1e-12
    percolation_water_volume_fraction: float = 0 
    conductivity_exponent: float = 0

    def __post_init__(self):
        """
        Compute derived properties of the membrane after initialization.
        """
        self.ionomer_concentration = self.dry_density / self.equivalent_weight  # kmol/m³
        self.ionomer_molar_volume = 1. / self.ionomer_concentration  # m³/kmol
    
    def wet_density(self, water_content, temperature):
            """
            Compute the wet density of the ionomer.

            Parameters
            ----------
            water_content : float
                Water content in the ionomer [n.d.].
            temperature : float
                Temperature [K].

            Returns
            -------
            float
                Wet density [kg/m3].
            """
            water_mass = water_molecular_weight * water_content
            return self.equivalent_weight + water_mass / (self.equivalent_weight / self.bulk_density + water_mass / water_density(temperature))

    def heat_of_adsorption(self, temperature):
        return enthalpy_condensation(temperature)
    
    def wet_expansion_factor(self, water_content, temperature):
        """
        Compute volumetric expansion factor due to water uptake.

        Parameters
        ----------
        water_content : float
            Water content [n.d.].
        temperature : float
            Temperature [K].

        Returns
        -------
        float
            Expansion factor relative to dry volume.
        """
        water_mass = water_molecular_weight * water_content
        return 1 + self.dry_density * water_mass / self.equivalent_weight / water_density(temperature)