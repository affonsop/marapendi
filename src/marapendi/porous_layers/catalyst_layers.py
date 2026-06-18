
"""
Module providing classes to model catalyst layers in electrochemical cells.
"""
from dataclasses import dataclass, field
import numpy as np
from ..thermo.constants import GAS_CONSTANT

from ..thermo.electrochemistry import ElectrochemicalReaction 
from ..membrane.ionomer import Ionomer
from ..membrane.pem import PFSAIonomer
from .porous_layers import PorousLayer
from ..thermo.water import o2_water_diffusivity 

@dataclass
class CatalystLayer(PorousLayer):
    """
    Represents a generic catalyst layer in a fuel cell or electrolyser, which includes
    an ionomer phase, catalyst particles, and electrochemical reaction sites.

    Attributes
    ----------
    ionomer : Ionomer
        Ionomer model associated with the catalyst layer.
    reaction : ElectrochemicalReaction
        Electrochemical reaction model.
    catalyst_loading : float
        Catalyst loading in g/cm² (default: 0.2 mg/cm²).
    ionomer_to_catalyst_ratio : float
        Ratio of ionomer mass to catalyst mass.
    catalyst_density : float
        Density of catalyst particles (kg/m³).
    ecsa : float
        Electrochemically active surface area (m²/g).
    ionomer_vol_fraction : float
        Volume fraction of ionomer in the layer.
    ionomer_film_thickness : float
        Thickness of the ionomer film around catalyst particles (m).
    contact_angle : float
        Contact angle (degrees).
    theta_catalyst : float
        Catalyst surface coverage (dimensionless).
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
        tort_ion = self.ionomer.tortuosity(self.ionomer_vol_fraction)
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
        ---------
        Neyerlin, K. C. et al. J. Electrochem. Soc. 154, B279 (2007).
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        self.sheet_resistance = 1. / (1. / self.ionomer_sheet_charge_resistance(ionomer_water_content, temperature, charge) +
                                      1. / self.electrolyte_sheet_resistance(temperature))
        nu = np.minimum(self.sheet_resistance * current_density / self.reaction.tafel_slope(temperature), 10)
        self.xi_neyerlin = nu * (-8.287e-3 * nu + 0.7184) - 2.072e-3
        return self.sheet_resistance / (3 + self.xi_neyerlin)

    def electrolyte_sheet_resistance(self, temperature, electrolyte_saturation=1.e-12):
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
        electrolyte_conductivity = self.electrolyte.calculate_ionic_conductivity(temperature)
        return self.thickness / ((electrolyte_saturation * self.porosity) ** 1.5 * electrolyte_conductivity) 
    
    def activation_overpotential(self, current_density, activity):
        """
        Compute activation overpotential based on current density.

        Parameters
        ----------
        current_density : float
            Current density [A/m²].
        activity : float
            Effective activity of reactant.

        Returns
        -------
        float
            Activation overpotential [V].
        """
        return self.reaction.tafel_overpotential(
            (current_density) / (self.ecsa * self.catalyst_loading),
            self.temperature,
            activity
        )
    
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
        """
        Calculate derived PTL properties such as fiber density,
        surface area, ionomer volume fraction and update porosity.
        """
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
    """
    Catalyst layer model with explicit platinum/carbon and ionomer structure.

    Attributes
    ----------
    platinum_loading : float
        Pt loading [g/cm²].
    catalyst_platinum_weight_percent : float
        Pt weight percent in catalyst.
    ionomer_to_carbon_ratio : float
        Ionomer/carbon mass ratio.
    platinum_density, carbon_density : float
        Densities [kg/m³].
    carbon_agglomerate_radius : float
        Radius of carbon agglomerates [m].
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
        """
        Compute volume fractions and geometry based on catalyst composition.

        Equations come from Hao et al. (2015). 

        References
        ----------
        Hao, L. et al. J. Electrochem. Soc. 162, F854 (2015).
        """
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
        """
        Update ionomer volume fraction, film thickness, and porosity for
        given water content and temperature.
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
        ionomer_radius = self.carbon_agglomerate_radius + self.ionomer_film_thickness 
        self.water_film_thickness = (water_saturation * self.porosity * self.carbon_agglomerate_radius ** 3 / self.carbon_vol_fraction + ionomer_radius ** 3 ) ** (1./3) - ionomer_radius

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
      