"""
Catalyst layer components: static physical properties and correlations.

:class:`CatalystLayer` and :class:`PtCCatalystLayer` extend
:class:`~marapendi.porous_layers.PorousLayer` with the
electrochemical-reaction and ionomer properties of a catalyst layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .electrochemistry import ElectrochemicalReaction
from .ionomer import CatalystLayerIonomer, PFSAIonomer
from .water import o2_water_diffusivity
from .constants import GAS_CONSTANT
from .porous_layers import PorousLayer


@dataclass
class CatalystLayer(PorousLayer):
    """Generic catalyst layer: an ionomer phase, catalyst particles and reaction sites.

    Attributes
    ----------
    ionomer : CatalystLayerIonomer
        Ionomer transport-property correlations.
    reaction : ElectrochemicalReaction
        Electrochemical reaction kinetics.
    catalyst_loading : float
        Catalyst loading (kg/m^2).
    ecsa : float
        Electrochemically active surface area (m^2/kg catalyst).
    """

    ionomer: CatalystLayerIonomer = field(default_factory=CatalystLayerIonomer)
    reaction: ElectrochemicalReaction = field(default_factory=ElectrochemicalReaction)
    catalyst_loading: float = 0.2e-6 * 1e4
    ecsa: float = 70e3


@dataclass
class PtCCatalystLayer(CatalystLayer):
    """Pt/C catalyst layer with explicit Pt/carbon/ionomer microstructure.

    Attributes
    ----------
    ionomer : PFSAIonomer
        Ionomer transport-property correlations.
    thickness : float
        Catalyst layer thickness (m).
    platinum_loading : float
        Platinum loading (kg/m^2).
    catalyst_platinum_weight_percent : float
        Platinum weight fraction in the catalyst (Pt/C).
    ionomer_to_carbon_ratio : float
        Dry ionomer-to-carbon mass ratio.
    platinum_density, carbon_density : float
        Densities of platinum and carbon (kg/m^3).
    carbon_agglomerate_radius : float
        Radius of the carbon agglomerates (m).
    contact_angle : float
        Contact angle of the wetting phase (degrees).
    omega_PtO : float
        Platinum-oxide coverage parameter.
    ionomer_k1, ionomer_k2, ionomer_k3 : float
        Empirical interfacial-resistance coefficients (Hao et al. 2015).
    """

    ionomer: PFSAIonomer = field(default_factory=PFSAIonomer)
    thickness: float = 10e-6
    platinum_loading: float = 0.2e-6 * 1e4
    catalyst_platinum_weight_percent: float = 0.5
    ionomer_to_carbon_ratio: float = 0.75
    platinum_density: float = 21450.
    carbon_density: float = 1950.
    carbon_agglomerate_radius: float = 25e-9
    contact_angle: float = 95.
    omega_PtO: float = 3000e3
    ionomer_k1: float = 8.5
    ionomer_k2: float = 5.4
    ionomer_k3: float = 5.4

    def __post_init__(self):
        super().__post_init__()

        carbon_loading = self.platinum_loading * (1 / self.catalyst_platinum_weight_percent - 1)
        self.carbon_vol_fraction = carbon_loading / (self.thickness * self.carbon_density)
        platinum_vol_fraction = self.platinum_loading / (self.thickness * self.platinum_density)
        self.catalyst_vol_fraction = platinum_vol_fraction + self.carbon_vol_fraction
        self.dry_ionomer_vol_fraction = (
            self.carbon_vol_fraction * self.carbon_density * self.ionomer_to_carbon_ratio / self.ionomer.dry_density
        )

        carbon_agglomerate_surface = 4 * np.pi * self.carbon_agglomerate_radius ** 2
        carbon_agglomerate_volume = carbon_agglomerate_surface * self.carbon_agglomerate_radius / 3.
        self.carbon_agglomerate_number_density = self.carbon_vol_fraction / carbon_agglomerate_volume

    def _ionomer_vol_fraction(self, ionomer_water_content: float, temperature: float) -> float:
        ionomer_expansion_factor = self.ionomer.wet_expansion_factor(
            np.clip(ionomer_water_content, 3, 20), temperature,
        )
        return self.dry_ionomer_vol_fraction * ionomer_expansion_factor

    def ionomer_vol_fraction(self, state) -> float:
        """Wet ionomer volume fraction, given the catalyst layer ``state``."""
        return self._ionomer_vol_fraction(state.ionomer_water_content, state.temperature)

    def ionomer_sheet_charge_resistance(
        self, ionomer_water_content: float, temperature: float, charge: str = 'proton',
    ) -> float:
        """Ionomer film charge resistance (Ohm.m^2), with tortuosity from Hao et al. (2016)."""
        ionomer_charge_conductivity = self.ionomer.charge_conductivity(ionomer_water_content, temperature, charge)
        eps_ion = self._ionomer_vol_fraction(ionomer_water_content, temperature)
        tort_ion = np.where(eps_ion > 0.16, 1, 0.0845 * (np.maximum(0.1, eps_ion) - 0.04) ** -1.17)
        return self.thickness / (eps_ion / tort_ion * ionomer_charge_conductivity)

    def effective_charge_resistance(
        self, current_density: float, ionomer_water_content: float, temperature: float, charge: str = 'proton',
    ) -> float:
        """Effective charge resistance (Ohm.m^2), from Neyerlin et al. (2007) / Goshtasbi et al. (2020)."""
        sheet_resistance = self.ionomer_sheet_charge_resistance(ionomer_water_content, temperature, charge)
        nu = np.minimum(sheet_resistance * current_density / self.reaction.tafel_slope(temperature), 10)
        xi_neyerlin = nu * (-8.287e-3 * nu + 0.7184) - 2.072e-3
        return sheet_resistance / (3 + xi_neyerlin)

    def wet_porosity(self, state) -> float:
        """Porosity of the wet catalyst layer."""
        return 1 - self.catalyst_vol_fraction - self.ionomer_vol_fraction(state)

    def ionomer_film_thickness(self, state) -> float:
        """Thickness of the ionomer film coating the carbon agglomerates (m)."""
        ionomer_vol_fraction = self.ionomer_vol_fraction(state)
        return self.carbon_agglomerate_radius * (
            (ionomer_vol_fraction / self.carbon_vol_fraction + 1) ** (1 / 3) - 1
        )

    def ionomer_vol_surface_area(self, state) -> float:
        """Ionomer/gas interfacial area per unit volume (m^2/m^3)."""
        ionomer_film_thickness = self.ionomer_film_thickness(state)
        return (
            4 * np.pi * (self.carbon_agglomerate_radius + ionomer_film_thickness) ** 2
            * self.carbon_agglomerate_number_density
        )

    def water_film_thickness(self, state) -> float:
        """Thickness of the liquid water film around the agglomerates (m)."""
        ionomer_radius = self.carbon_agglomerate_radius + self.ionomer_film_thickness(state)
        porosity = self.wet_porosity(state)
        return (
            state.liquid_saturation * porosity * self.carbon_agglomerate_radius ** 3 / self.carbon_vol_fraction
            + ionomer_radius ** 3
        ) ** (1. / 3) - ionomer_radius

    def o2_ionomer_film_bulk_resistance(self, state) -> float:
        """Bulk O2 transport resistance through the ionomer film (s/m)."""
        ionomer_film_thickness = self.ionomer_film_thickness(state)
        return ionomer_film_thickness / (
            GAS_CONSTANT * state.temperature
            * self.ionomer.o2_permeability(state.ionomer_water_content, state.temperature)
        )

    def o2_ionomer_film_resistance(self, state) -> float:
        """Total O2 transport resistance through the ionomer/water film (s/m).

        Interface resistances from eq. 32 of Hao et al. (2015), neglecting
        the effect of the water film on the interface terms.
        """
        ionomer_vol_surface_area = self.ionomer_vol_surface_area(state)
        water_film_thickness = self.water_film_thickness(state)

        ionomer_pt_interface_term = (
            (self.ionomer_k2 + 1) / (1 - state.theta_catalyst) / (self.platinum_loading * self.ecsa)
        )
        ionomer_gas_interface_term = self.ionomer_k1 / (ionomer_vol_surface_area * self.thickness)
        water_term = (
            (self.ionomer_k3 + 1) * water_film_thickness / o2_water_diffusivity(state.temperature)
            / (ionomer_vol_surface_area * self.thickness)
        )
        return (
            (ionomer_gas_interface_term + ionomer_pt_interface_term)
            * self.o2_ionomer_film_bulk_resistance(state)
            + water_term
        )
