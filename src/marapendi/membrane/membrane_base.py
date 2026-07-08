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
from .ionomer_base import Ionomer


@dataclass
class Membrane:
    """Static properties of a proton/anion exchange membrane.

    Holds membrane geometry and delegates transport correlations to the
    composed :attr:`ionomer`.  
    
    Attributes
    ----------
    dry_thickness : float
        Membrane thickness in the dry state (m).
    ionomer : Ionomer
        Ionomer instance providing transport correlations.
    volume_heat_capacity : float
        Volumetric heat capacity in J/(m3.K).
    """

    dry_thickness: float = 25e-6
    ionomer: Ionomer = field(default=None)
    volume_heat_capacity: float = 1e6 

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
        """
        return self.ionomer.h2_permeability(water_content, temperature) * h2_pressure_difference / self.dry_thickness

    def charge_resistance(self, water_content, temperature, charge='proton'):
        """Through-plane charge resistance (Ω·m²)."""
        return self.dry_thickness / self.ionomer.charge_conductivity(water_content, temperature, charge)
