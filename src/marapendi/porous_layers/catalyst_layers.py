
"""
Catalyst layer models for electrochemical cells.

:class:`CatalystLayer` is the base class, providing ionomer charge resistance,
effective charge resistance (Neyerlin / Goshtasbi distribution model), and a
Bruggeman electrolyte resistance for alkaline layers.

:class:`PtCCatalystLayer` extends it with an explicit Pt/C agglomerate geometry
following Hao et al. (2015): volume fractions are computed from platinum and
carbon loadings, the wet ionomer film thickness is updated at each operating
point, and local O₂ transport resistance across the ionomer film (bulk + gas/Pt
interface terms) is evaluated via :meth:`~PtCCatalystLayer.o2_ionomer_film_resistance`.

:class:`PorousTransferLayer` adapts the geometry to a fibrous PTL structure.
"""
from dataclasses import dataclass, field
import numpy as np
from ..thermo.constants import GAS_CONSTANT

from ..thermo.electrochemistry import ElectrochemicalReaction 
from ..membrane.ionomer_base import Ionomer
from ..membrane.pem import PFSAIonomer
from .porous_layers import PorousLayer
from ..thermo.water import o2_water_diffusivity 

@dataclass
class CatalystLayer(PorousLayer):
    """Base class for catalyst layers in fuel cells and electrolysers.

    Combines a porous electrode structure with an ionomer phase and an
    electrochemical reaction model.  Provides ionomer/electrolyte charge
    resistance and activation overpotential; subclasses add geometry-specific
    transport models.

    Attributes
    ----------
    ionomer : Ionomer
        Ionomer transport model (proton/hydroxide conductivity, O₂ permeability).
    reaction : ElectrochemicalReaction
        Electrochemical reaction parameters (exchange current density, Tafel slope).
    catalyst_loading : float
        Total catalyst loading (kg/m²).
    ionomer_to_catalyst_ratio : float
        Ionomer/catalyst mass ratio (–).
    catalyst_density : float
        Catalyst particle density (kg/m³).
    ecsa : float
        Electrochemically active surface area (m²/kg).
    ionomer_vol_fraction : float
        Volume fraction of ionomer (–).
    ionomer_film_thickness : float
        Ionomer film thickness around catalyst particles (m).
    contact_angle : float
        Contact angle between liquid water and the solid phase (°).
    """
    ionomer: Ionomer = field(default_factory=PFSAIonomer)
    reaction: ElectrochemicalReaction = field(default_factory=ElectrochemicalReaction)
    catalyst_loading: float = 0.2e-6 * 1e4
    ionomer_to_catalyst_ratio: float = 0.75
    catalyst_density: float = 21450
    ecsa: float = 70e3
    ionomer_vol_fraction: float = 0.3
    ionomer_film_thickness: float = 2e-9
    contact_angle: float = 95.

    def ionomer_sheet_charge_resistance(self, ionomer_water_content, temperature, charge='proton'):
        """
        Compute ionomer film proton resistance.

        Parameters
        ----------
        ionomer_water_content : float
            Water content.
        temperature : float
            Temperature in Kelvin.
        charge : str
            Charge carrier.

        Returns
        -------
        float
            Ionomer resistance [Ohm.m²].
        
        References
        ----------
        Hao, L. et al. J. Electrochem. Soc. 163, F744 (2016).
        """
        ionomer_charge_conductivity = self.ionomer.charge_conductivity(ionomer_water_content, temperature, charge)
        eps_ion = self.ionomer_vol_fraction
        tort_ion = self.ionomer_tortuosity(self.ionomer_vol_fraction)
        return self.thickness / (eps_ion / tort_ion * ionomer_charge_conductivity)


    def effective_charge_resistance(self, current_density, ionomer_water_content, temperature, charge='proton'):
        """
        Calculate effective charge resistance using the equations from Goshtasbi et al. (2020)
        based on Neyerlin et al. (2007).

        Parameters
        ----------
        current_density : float
            Current density [A/m²].
        ionomer_water_content : float
            Water content in the ionomer.
        temperature : float
            Temperature in Kelvin.
        charge : str
            Charge carrier type.

        Returns
        -------
        float
            Effective charge resistance [Ohm.m²].
        
        References
        ----------
        Neyerlin, K. C. et al. J. Electrochem. Soc. 154, B279 (2007).
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        self.sheet_resistance = 1. / (1. / self.ionomer_sheet_charge_resistance(ionomer_water_content, temperature, charge) +
                                      1. / self.electrolyte_sheet_resistance(temperature))
        nu = np.minimum(self.sheet_resistance * current_density / self.reaction.tafel_slope(temperature), 10)
        self.xi_neyerlin = nu * (-8.287e-3 * nu + 0.7184) - 2.072e-3
        return self.sheet_resistance / (3 + self.xi_neyerlin)

    def electrolyte_sheet_resistance(self, temperature, electrolyte_saturation=0):
        """
        Calculate electrolyte sheet resistance in the porous electrolyte phase.
        Uses Bruggeman correlation for electrolyte. 

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        electrolyte_saturation : float
            Electrolyte saturation.
        Returns
        -------
        float
            Electrolyte sheet resistance [Ohm.m²].
        """
        electrolyte_saturation = np.maximum(electrolyte_saturation, 1e-12)
        electrolyte_conductivity = self.electrolyte.calculate_ionic_conductivity(temperature)
        return self.thickness / ((electrolyte_saturation * self.porosity) ** 1.5 * electrolyte_conductivity) 

    def ionomer_tortuosity(self, volume_fraction: float) -> float:
        """Bruggeman tortuosity factor for the ionomer phase.

        Standard Bruggeman inverse power-law approximation for a randomly
        distributed phase with exponent -0.5.

        Parameters
        ----------
        volume_fraction : float
            Ionomer volume fraction (n.d.).

        Returns
        -------
        float
            Tortuosity factor (n.d.).
        """
        return volume_fraction ** -0.5
    
    
@dataclass
class PorousTransferLayer(CatalystLayer):
    """
    Represents a porous transfer layer (PTL) over a catalyst layer,
    e.g. fibrous media structure in fuel cells.

    Attributes
    ----------
    ptl_porosity : float
        PTL porosity (default: 0.83).
    fiber_diameter : float
        Diameter of fibers [m].
    ionomer_k1, ionomer_k2 : float
        Constants from Hao et al. (2015) for film resistance calculations.
    """
    ptl_porosity: float = 0.83
    fiber_diameter: float = 20e-6
    ionomer_k1: float = 8.5
    ionomer_k2: float = 5.4

    def __post_init__(self):
        """Compute derived PTL geometry from fiber and catalyst composition."""
        self.fiber_number_density_cross_section = 4 * (1 - self.ptl_porosity) / (np.pi * self.fiber_diameter ** 2)
        self.fiber_surface_per_volume = 4 * (1 - self.ptl_porosity) / self.fiber_diameter
        self.catalyst_vol_surface_area = self.catalyst_loading * self.ecsa / self.thickness
        self.catalyst_vol_fraction = self.catalyst_loading / (self.thickness * self.catalyst_density)
        self.ionomer_vol_fraction = (self.catalyst_vol_fraction * self.catalyst_density *
                                     self.ionomer_to_catalyst_ratio / self.ionomer.dry_density)
        self.porosity = self.ptl_porosity - self.catalyst_vol_fraction - self.ionomer_vol_fraction
        self.ionomer_film_thickness = self.ionomer_vol_fraction / self.fiber_surface_per_volume
        self.ionomer_vol_surface_area = (np.pi *
                                         (self.fiber_diameter + 2 * self.ionomer_film_thickness) *
                                         self.fiber_number_density_cross_section)
        self.effective_gas_diffusion_ratio = self.porosity ** 1.5
        PorousLayer.__post_init__(self)




@dataclass
class PtCCatalystLayer(CatalystLayer):
    """Catalyst layer with explicit Pt/C agglomerate geometry and ionomer film.

    Volume fractions (carbon, platinum, ionomer, pore) are derived from the
    catalyst composition at construction time.  The wet ionomer film thickness
    is updated at each operating point via :meth:`set_ionomer_wet_properties`.

    Local O₂ transport resistance across the ionomer film is modelled following
    Hao et al. (2015): bulk diffusion through the film plus gas/ionomer and
    ionomer/Pt interface resistances weighted by empirical factors ``ionomer_k1``
    (gas-side) and ``ionomer_k2`` (Pt-side).  A water-film correction term
    (``ionomer_k3``) accounts for flooding at the ionomer surface.

    Attributes
    ----------
    platinum_loading : float
        Pt loading (kg/m²).
    catalyst_platinum_weight_percent : float
        Pt weight fraction in the catalyst powder (–).
    ionomer_to_carbon_ratio : float
        Ionomer/carbon mass ratio (–).
    platinum_density, carbon_density : float
        Material densities (kg/m³).
    carbon_agglomerate_radius : float
        Radius of carbon agglomerates (m).
    ionomer_k1, ionomer_k2, ionomer_k3 : float
        Interface resistance pre-factors from Hao et al. (2015).
    omega_PtO : float
        Molar volume of platinum oxide (m³/kmol), used by the PtO coverage model.

    References
    ----------
    Hao, L. et al. J. Electrochem. Soc. 162, F854 (2015).
    """
    porosity: float = None
    platinum_loading: float = 0.2e-6 * 1e4
    catalyst_platinum_weight_percent: float = 0.5
    ionomer_to_carbon_ratio: float = 0.75
    platinum_density: float = 21450
    carbon_density: float = 1950
    platinum_vol_surface_area: float = 0
    carbon_agglomerate_radius: float = 25e-9
    dry_ionomer_vol_fraction: float = 0
    ionomer_vol_fraction: float = 0
    ionomer_film_thickness: float = 0
    contact_angle: float = 95.
    omega_PtO: float = 3000e3
    ionomer_k1: float = 8.5
    ionomer_k2: float = 5.4
    ionomer_k3: float = 5.4
    ionomer_water_content: float = 10

    def __post_init__(self):
        """Compute volume fractions and geometry from catalyst composition (Hao et al. 2015)."""
        if self.platinum_vol_surface_area == 0:
            self.platinum_vol_surface_area = self.platinum_loading * self.ecsa / self.thickness
        if not self.porosity:
            self.carbon_loading = self.platinum_loading * (1 / self.catalyst_platinum_weight_percent - 1)
            self.carbon_vol_fraction = self.carbon_loading / (self.thickness * self.carbon_density)
            self.platinum_vol_fraction = self.platinum_loading / (self.thickness * self.platinum_density)
            self.catalyst_vol_fraction = self.platinum_vol_fraction + self.carbon_vol_fraction
            self.dry_ionomer_vol_fraction = (self.carbon_vol_fraction * self.carbon_density *
                                             self.ionomer_to_carbon_ratio / self.ionomer.dry_density)
        else:
            self.catalyst_vol_fraction = 1 - self.dry_ionomer_vol_fraction - self.porosity

        self.carbon_agglomerate_surface = 4 * np.pi * self.carbon_agglomerate_radius ** 2
        self.carbon_agglomerate_volume = self.carbon_agglomerate_surface * self.carbon_agglomerate_radius / 3.
        self.carbon_agglomerate_number_density = self.carbon_vol_fraction / self.carbon_agglomerate_volume
        self.set_ionomer_wet_properties(self.ionomer_water_content, 300)
        self.set_water_film_thickness(0)
        PorousLayer.__post_init__(self)

    def set_ionomer_wet_properties(self, ionomer_water_content, temperature):
        """Update ionomer volume fraction, film thickness, and pore volume for given water content.

        Computes the wet ionomer expansion factor from the ionomer model, then
        recalculates ``ionomer_vol_fraction``, ``porosity``, ``ionomer_film_thickness``,
        ``ionomer_vol_surface_area``, and ``effective_gas_diffusion_ratio``.
        Called at construction and at every MEA temperature update.
        """
        self.ionomer_expansion_factor = self.ionomer.wet_expansion_factor(np.clip(ionomer_water_content, 3, 20), temperature)
        self.ionomer_vol_fraction = self.dry_ionomer_vol_fraction * self.ionomer_expansion_factor
        self.porosity = 1 - self.catalyst_vol_fraction - self.ionomer_vol_fraction
        self.ionomer_film_thickness = self.carbon_agglomerate_radius * (
            (self.ionomer_vol_fraction / self.carbon_vol_fraction + 1) ** (1/3) - 1)
        self.ionomer_vol_surface_area = 4 * np.pi * (self.carbon_agglomerate_radius + self.ionomer_film_thickness) ** 2 * self.carbon_agglomerate_number_density
        self.ionomer_to_carbon_vol_ratio = (1 + self.ionomer_film_thickness / self.carbon_agglomerate_radius)**3 - 1
        self.effective_gas_diffusion_ratio = self.porosity ** 1.5
    
    def set_water_film_thickness(self, water_saturation):
        """Compute and store the liquid water film thickness around agglomerates.

        Parameters
        ----------
        water_saturation : float
            Non-wetting (liquid water) saturation in the pore space (–).
        """
        ionomer_radius = self.carbon_agglomerate_radius + self.ionomer_film_thickness
        self.water_film_thickness = (
            water_saturation * self.porosity * self.carbon_agglomerate_radius ** 3
            / self.carbon_vol_fraction + ionomer_radius ** 3
        ) ** (1. / 3) - ionomer_radius

    def o2_ionomer_film_bulk_resistance(self, ionomer_water_content, temperature):
        """
        Calculate the oxygen bulk resistance across the ionomer film.

        Parameters
        ----------
        ionomer_water_content : float
            Water content in the ionomer.
        temperature : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Oxygen film resistance [s/m].
        """
        return (self.ionomer_film_thickness /
                (GAS_CONSTANT * temperature * self.ionomer.o2_permeability(ionomer_water_content, temperature)))

    def o2_ionomer_film_resistance(self, ionomer_water_content, temperature, catalyst_availability=1):
        """
        Calculate total oxygen film resistance. 
        Interface resistances are calculated according to equation 32 in Hao et al. (2015),
        but neglecting the effect of the water film. 

        Parameters
        ----------
        ionomer_water_content : float
            Water content in ionomer.
        temperature : float
            Temperature in Kelvin.
        catalyst_availability: float
            Catalyst availability = 1 - catalyst coverage ratio. 
            
        Returns
        -------
        float
            Total oxygen film resistance [s/m].

        References
        ----------
        Hao, L. et al. J. Electrochem. Soc. 162, F854 (2015).
        """
        ionomer_pt_interface_term = (self.ionomer_k2 + 1) / (self.platinum_loading * self.ecsa * catalyst_availability)
        ionomer_gas_interface_term = self.ionomer_k1 / (self.ionomer_vol_surface_area * self.thickness)
        water_term = (self.ionomer_k3 + 1) * self.water_film_thickness / o2_water_diffusivity(temperature) / (self.ionomer_vol_surface_area * self.thickness)
        return (ionomer_gas_interface_term + ionomer_pt_interface_term) * self.o2_ionomer_film_bulk_resistance(ionomer_water_content, temperature) + water_term
    
    def ionomer_tortuosity(self, volume_fraction: float) -> float:
        """Piecewise tortuosity factor for the ionomer film in a Pt/C agglomerate layer.

        Above a threshold of 0.16 the ionomer network is well-connected and the
        tortuosity approaches 1.  Below that threshold a fitted power law is used,
        following the correlation from Hao et al. (2015).

        Parameters
        ----------
        volume_fraction : float
            Ionomer volume fraction (–).

        Returns
        -------
        float
            Tortuosity factor (–).

        References
        ----------
        Hao, L. et al. J. Electrochem. Soc. 162, F854 (2015).
        """
        return np.where(
            volume_fraction > 0.16,
            1,
            0.0845 * (np.maximum(0.1, volume_fraction) - 0.04) ** -1.17,
        )