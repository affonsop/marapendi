"""
Membrane components: static physical properties and correlations.

A :class:`Membrane` holds the geometric properties of the membrane (dry
thickness) and delegates all transport correlations to a composed
:class:`~marapendi.ionomer.Ionomer` instance.  Specialisations such as
:class:`~marapendi.membrane.pem.PFSA` swap in a concrete ionomer subclass and
add membrane-specific sorption/conductivity correlations.

The composition replaces the previous inheritance from
:class:`~marapendi.ionomer.Ionomer`: ionomer properties are accessed via
``membrane.ionomer.xxx`` or through the thin delegation methods provided here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from marapendi.thermo.water import water_molar_volume
from marapendi.membrane.permeation import HydrogenPermeationModel
from .ionomer_base import Ionomer


@dataclass
class Membrane:
    """Static properties of a proton/anion exchange membrane.

    Holds membrane geometry and delegates transport correlations to the
    composed :attr:`ionomer`.  The optional :attr:`h2_permeation_model`
    overrides the default ionomer-based H2 crossover correlation.

    Attributes
    ----------
    dry_thickness : float
        Membrane thickness in the dry state (m).
    ionomer : Ionomer
        Ionomer instance providing transport correlations.
    h2_permeation_model : HydrogenPermeationModel or None
        Optional custom H2 permeation model.  When ``None`` the ionomer's
        ``h2_permeability`` is used directly.
    """

    dry_thickness: float = 25e-6
    ionomer: Ionomer = field(default=None)
    h2_permeation_model: HydrogenPermeationModel = field(default=None)

    def __post_init__(self):
        if self.ionomer is not None:
            self.surface_concentration = self.ionomer.dry_concentration * self.dry_thickness

    # ------------------------------------------------------------------
    # Ionomer delegation — methods called by water_balance.py
    # ------------------------------------------------------------------

    @property
    def dry_concentration(self) -> float:
        """Dry molar concentration of ion-exchange sites (kmol/m³)."""
        return self.ionomer.dry_concentration

    def calculate_water_diffusivity(self, temperature: float) -> float:
        """Adsorbed-water diffusivity (m²/s) — delegates to ionomer."""
        return self.ionomer.calculate_water_diffusivity(temperature)

    def calculate_water_absorption_coefficient(self, temperature: float) -> float:
        """Water absorption coefficient (m/s) — delegates to ionomer."""
        return self.ionomer.calculate_water_absorption_coefficient(temperature)

    def calculate_electroosmotic_drag_speed(self, temperature: float, current_density: float) -> float:
        """Electroosmotic drag velocity (m/s) — delegates to ionomer."""
        return self.ionomer.calculate_electroosmotic_drag_speed(temperature, current_density)

    # ------------------------------------------------------------------
    # Membrane-level methods
    # ------------------------------------------------------------------

    def hydrogen_permeation_flux(
        self,
        h2_pressure_difference: float,
        temperature: float,
        water_content: float,
    ) -> float:
        """Hydrogen permeation flux through the membrane (kmol/m²/s).

        Uses :attr:`h2_permeation_model` when set; otherwise falls back to
        the ionomer's ``h2_permeability`` correlation.
        """
        if self.h2_permeation_model is not None:
            water_vol_fraction = self.ionomer.water_vol_fraction(
                water_content, water_molar_volume(temperature)
            )
            return self.h2_permeation_model.permeation_flux(
                self.dry_thickness, h2_pressure_difference, temperature, 0., water_vol_fraction
            )
        return self.ionomer.h2_permeability(water_content, temperature) * h2_pressure_difference / self.dry_thickness

    def charge_resistance(self, water_content, temperature, charge='proton'):
        """Through-plane charge resistance (Ω·m²)."""
        return self.dry_thickness / self.ionomer.charge_conductivity(water_content, temperature, charge)
