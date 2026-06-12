"""
Ionomer material properties and correlations.

:class:`Ionomer` holds the dry material properties of an ion-exchange
polymer (equivalent weight, density, water-transport correlations) and
the correlation methods that depend only on those properties together
with ``water_content``/``temperature``.
:class:`~marapendi.future.membrane.Membrane` inherits from it and adds
membrane-specific geometry (thickness, hydrogen permeation, ...).
"""
from __future__ import annotations

from dataclasses import dataclass

from marapendi.tools.tools import arrhenius_term
from .constants import FARADAY_CONSTANT
from ..legacy.water import water_density, water_molecular_weight


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

    equivalent_weight: float = 1.1e3
    dry_density: float = 1980.

    reference_water_diffusivity: float = 4.3e-10
    reference_absorption_coefficient: float = 1e-5
    reference_temperature: float = 353.15
    water_diffusivity_activation_energy: float = 20e6
    water_absorption_activation_energy: float = 20e6

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
            self.water_diffusivity_activation_energy, temperature, self.reference_temperature,
        )

    def calculate_water_absorption_coefficient(self, temperature: float) -> float:
        """Water absorption coefficient (m/s) at ``temperature``."""
        return self.reference_absorption_coefficient * arrhenius_term(
            self.water_absorption_activation_energy, temperature, self.reference_temperature,
        )

    def calculate_electroosmotic_drag_coefficient(self, temperature: float, water_content: float) -> float:
        """Electroosmotic drag coefficient (n.d.) for a given ``water_content``."""
        return (0.02 * temperature - 3.86) / 22.5 * water_content

    def calculate_electroosmotic_drag_speed(self, temperature: float, current_density: float) -> float:
        """Electroosmotic drag speed (m/s) for a given ``current_density`` (A/m^2)."""
        return (
            self.calculate_electroosmotic_drag_coefficient(temperature, 1)
            * current_density / FARADAY_CONSTANT / self.dry_concentration
        )
