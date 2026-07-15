"""
Ionomer material properties and correlations.

:class:`Ionomer` is the base class for ion-exchange polymer models.  It
holds dry material properties (equivalent weight, density) and Arrhenius
correlations for water diffusivity, water absorption, and electroosmotic
drag.

:class:`~marapendi.components.membrane.membrane_base.Membrane` *composes* an ``ionomer`` instance
rather than inheriting from :class:`Ionomer`.  Membrane-level geometry
(dry thickness, H₂ permeation) and water-balance models sit on
:class:`~marapendi.components.membrane.membrane_base.Membrane`; ionomer correlations are accessed
via ``membrane.ionomer.<method>``, or through thin delegation helpers
defined on :class:`~marapendi.components.membrane.membrane_base.Membrane`.

:class:`~marapendi.components.membrane.pem.PFSAIonomer` and
:class:`~marapendi.components.membrane.aem.PAPIonomer` are concrete subclasses with
charge-transport correlations used by both membranes and catalyst layers.
A :class:`~marapendi.components.porous_layers.catalyst_layers.CatalystLayer` holds an
``ionomer: Ionomer`` field directly — no separate ``CatalystLayerIonomer``
subclass is required.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ...tools import arrhenius_term
from ...models.thermo.constants import FARADAY_CONSTANT, GAS_CONSTANT
from ...models.thermo.water import water_density, water_molar_volume, water_molecular_weight


@dataclass
class Ionomer:
    """Dry material properties of an ion-exchange polymer.

    Attributes
    ----------
    equivalent_weight : float
        Equivalent weight (kg/kmol).
    dry_density : float
        Dry density (kg/m^3).
    reference_water_diffusivity : float
        Reference adsorbed-water diffusivity (m^2/s).
    reference_absorption_coefficient : float
        Reference water absorption coefficient (m/s).
    reference_temperature : float
        Reference temperature for the Arrhenius corrections (K).
    water_diffusivity_activation_energy, water_absorption_activation_energy : float
        Activation energies for water diffusivity / absorption (J/kmol).
    """

    equivalent_weight: float 
    dry_density: float
    vapor_equilibrium_polynomial: np.ndarray
    reference_conductivity: float
    reference_water_diffusivity: float
    reference_water_absorption_coefficient: float  
    conductivity_exp: float
    conductivity_fv_threshold: float
    conductivity_activation_energy: float
    water_diffusivity_activation_energy: float
    water_absorption_activation_energy: float
    reference_conductivity_temperature: float
    reference_water_absorption_temperature: float
    reference_water_diffusivity_temperature: float
    

    def __post_init__(self):
        self.dry_concentration = self.dry_density / self.equivalent_weight  # kmol/m^3
        self.dry_molar_volume = 1. / self.dry_concentration  # m^3/kmol

    def water_vol_fraction(self, water_content: float, water_molar_volume: float) -> float:
        """Volume fraction of water in the ionomer, for a given ``water_content`` (mol H2O / mol SO3-)."""
        ionomer_water_molar_volume = water_molar_volume * water_content
        return ionomer_water_molar_volume / (self.dry_molar_volume + ionomer_water_molar_volume)

    def wet_density(self, water_content: float, temperature: float) -> float:
        """Wet density of the ionomer (kg/m^3)."""
        water_mass = water_molecular_weight * water_content
        return (
            (self.equivalent_weight + water_mass)
            / (self.equivalent_weight / self.dry_density + water_mass / water_density(temperature))
        )

    def wet_expansion_factor(self, water_content: float, temperature: float) -> float:
        """Volumetric expansion factor due to water uptake, relative to the dry volume."""
        water_mass = water_molecular_weight * water_content
        return 1 + self.dry_density * water_mass / self.equivalent_weight / water_density(temperature)

    def calculate_water_diffusivity(self, temperature: float) -> float:
        """Adsorbed water diffusivity (m^2/s) at ``temperature``."""
        return self.reference_water_diffusivity * arrhenius_term(
            self.water_diffusivity_activation_energy, temperature, self.reference_water_diffusivity_temperature,
        )

    def calculate_water_absorption_coefficient(self, temperature: float) -> float:
        """Water absorption coefficient (m/s) at ``temperature``."""
        return self.reference_water_absorption_coefficient * arrhenius_term(
            self.water_absorption_activation_energy, temperature, self.reference_water_absorption_temperature,
        )

    def calculate_electroosmotic_drag_coefficient(self, temperature: float, water_content: float) -> float:
        """Electroosmotic drag coefficient (n.d.) for a given ``water_content``."""
        return 1.

    def calculate_electroosmotic_drag_speed(self, temperature: float, current_density: float) -> float:
        """Electroosmotic drag speed (m/s) for a given ``current_density`` (A/m^2)."""
        return (
            self.calculate_electroosmotic_drag_coefficient(temperature, 1)
            * current_density / FARADAY_CONSTANT / self.dry_concentration
        )

    def charge_conductivity(self, water_content: float, temperature: float, charge: str = 'proton') -> float:
        """Charge conductivity (proton or hydroxide), in S/m.

        Dispatches to :meth:`proton_conductivity` or :meth:`hydroxide_conductivity`,
        which must be implemented by concrete subclasses.
        """
        if charge == 'proton':
            return self.proton_conductivity(water_content, temperature)
        elif charge == 'hydroxide':
            return self.hydroxide_conductivity(water_content, temperature)
        raise ValueError(f"Unknown charge carrier: {charge!r}")

