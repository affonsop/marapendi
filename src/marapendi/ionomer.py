"""
Ionomer material properties and correlations.

:class:`Ionomer` holds the dry material properties of an ion-exchange
polymer (equivalent weight, density, water-transport correlations) and
the correlation methods that depend only on those properties together
with ``water_content``/``temperature``.
:class:`~marapendi.membrane.Membrane` inherits from it and adds
membrane-specific geometry (thickness, hydrogen permeation, ...).

:class:`CatalystLayerIonomer` and its specializations (:class:`PFSAIonomer`,
:class:`PAPIonomer`) extend :class:`Ionomer` with the charge-transport
correlations needed by a catalyst layer's ionomer phase.
"""
from __future__ import annotations

import numpy as np
import cantera as ct
from dataclasses import dataclass

from marapendi.tools.tools import arrhenius_term
from .constants import FARADAY_CONSTANT
from .water import water_density, water_molar_volume, water_molecular_weight


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


@dataclass
class CatalystLayerIonomer(Ionomer):
    """Base class for the ionomer phase of a catalyst layer.

    Attributes
    ----------
    dry_density : float
        Dry density of the ionomer (kg/m^3).
    equivalent_weight : float
        Equivalent weight of the ionomer (kg/kmol).
    """

    dry_density: float = 2004.
    equivalent_weight: float = 952.

    def charge_conductivity(self, water_content: float, temperature: float, charge: str = 'proton') -> float:
        """Charge conductivity (proton or hydroxide), in S/m."""
        if charge == 'proton':
            return self.proton_conductivity(water_content, temperature)
        elif charge == 'hydroxide':
            return self.hydroxide_conductivity(water_content, temperature)


@dataclass
class PFSAIonomer(CatalystLayerIonomer):
    """PFSA ionomer (e.g. Nafion) with empirical fits for proton conductivity and O2 transport."""

    conductivity_correction: float = 1
    conductivity_exp: float = 1.5
    hydrated_proton_conductivity: float = 11
    conductivity_activation_energy: float = 11e6
    hydrated_o2_diffusion: float = 1.14698e-10 * 14 ** 0.708
    o2_diffusion_exponent: float = 0.708
    o2_diffusion_activation_energy: float = 24e6

    def o2_film_diffusion_coefficient(self, water_content: float, temperature: float = 353.15) -> float:
        """Effective O2 diffusion coefficient in the hydrated ionomer film (m^2/s)."""
        return (
            self.hydrated_o2_diffusion * (water_content / 14) ** self.o2_diffusion_exponent
            * arrhenius_term(self.o2_diffusion_activation_energy, temperature, 353.15)
        )

    def o2_permeability(self, water_content: float, temperature: float = 353.15) -> float:
        """O2 permeability (kmol/m/s/Pa) from a volume-fraction approach.

        Data from Goshtasbi et al. (2020).

        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        fv = self.water_vol_fraction(water_content, water_molar_volume(temperature))
        RT = ct.gas_constant * temperature
        return (6.74e-12 * np.exp(-21280e3 / RT) + fv * 50.5e-12 * np.exp(-20470e3 / RT)) * 1e-3

    def proton_conductivity(self, water_content: float, temperature: float) -> float:
        """Proton conductivity from empirical fits (S/m)."""
        fv = self.water_vol_fraction(water_content, water_molar_volume(temperature))
        return (
            self.conductivity_correction * 50 * (np.maximum(fv, 0.11) - 0.1) ** self.conductivity_exp
            * arrhenius_term(self.conductivity_activation_energy, temperature, 298.15)
        )

    def hydroxide_conductivity(self, water_content: float, temperature: float) -> float:
        """Hydroxide conductivity (S/m). PFSA ionomers do not conduct hydroxide."""
        return 1e-6

    def equilibrium_water_content(self, rh: float) -> float:
        """Equilibrium water content as a function of relative humidity.

        From Jinnouchi et al. (2021), fig S3a.

        References
        ----------
        Jinnouchi, R. et al. Nat. Commun. 12, 4956 (2021).
        """
        return 21.669 * rh ** 3 - 27.692 * rh ** 2 + 17.624 * rh + 0.688


NafionD2020 = PFSAIonomer(dry_density=2004., equivalent_weight=952.)


@dataclass
class PAPIonomer(CatalystLayerIonomer):
    """Poly(aryl piperidinium) (PAP) ionomer.

    Attributes
    ----------
    dry_density : float
        Density of the ionomer (kg/m^3). Data from Luo et al. (2020), table 1.
    equivalent_weight : float
        Equivalent weight of the ionomer (kg/kmol).

    References
    ----------
    Eon Chae, J. et al. J. Ind. Eng. Chem. 133, 255-262 (2024)
    Luo, X. et al. J. Memb. Sci. 598, 117680 (2020)
    Khalid, H. et al. Membranes (Basel) 12, 989 (2022).
    """

    dry_density: float = 1220.
    equivalent_weight: float = 1000 / 2.35

    def hydroxide_conductivity(self, water_content: float, temperature: float) -> float:
        """Hydroxide conductivity (S/m).

        Room-temperature conductivity for liquid-equilibrated PAP-20 from Luo
        et al. (2020), with activation energy from Khalid et al. (2022).

        References
        ----------
        Luo, X. et al. J. Memb. Sci. 598, 117680 (2020)
        Khalid, H. et al. Membranes (Basel) 12, 989 (2022).
        """
        return 5.8 * arrhenius_term(activation_energy=22.5e6, temperature=temperature, reference_temperature=298.15)
