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

from .tools import arrhenius_term
from .water import water_molecular_weight, water_molar_volume, water_density
from .membrane import Membrane, PAP85, PFSA

@dataclass 
class CatalystLayerIonomer(Membrane):
    """
    Base class for ionomer in a catalyst layer.

    Parameters
    ----------
    dry_density : float
        Dry density of the ionomer [kg/m3].
    equivalent_weight : float
        Equivalent weight of the ionomer [kg/kmol].
    """
    dry_density: float = 2004
    equivalent_weight: float = 952.

    def __post_init__(self):
        super().__post_init__()

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
        return self.equivalent_weight + water_mass / (self.equivalent_weight / self.dry_density + water_mass / water_density(temperature))

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

    def charge_conductivity(self, water_content, temperature, charge='proton'):
        """
        Compute charge conductivity (proton or hydroxide).

        Parameters
        ----------
        water_content : float
            Water content.
        temperature : float
            Temperature [K].
        charge : str, optional
            'proton' or 'hydroxide'. Default is 'proton'.

        Returns
        -------
        float
            Charge conductivity [S/m].
        """
        if charge == 'proton':
            return self.proton_conductivity(water_content, temperature)
        elif charge == 'hydroxide':
            return self.hydroxide_conductivity(water_content, temperature)

    def charge_resistance(self, water_content, temperature, charge='proton'):
        """
        Compute through-plane charge transport resistance.

        Returns
        -------
        float
            Charge transport resistance [Ohm.m2].
        """
        return self.dry_thickness / self.charge_conductivity(water_content, temperature, charge)

@dataclass
class PFSAIonomer(CatalystLayerIonomer):
    """
    PFSA ionomer (e.g. Nafion) with empirical fits for proton conductivity and O2 transport.
    """
    conductivity_correction: float = 1
    conductivity_exp: float = 1.5
    hydrated_proton_conductivity: float = 11
    conductivity_activation_energy: float = 11e6
    hydrated_o2_diffusion: float = 1.14698e-10 * 14 ** 0.708
    o2_diffusion_exponent: float = 0.708
    o2_diffusion_activation_energy: float = 24e6

    def o2_film_diffusion_coefficient(self, water_content, temperature=353.15):
        """
        Effective O2 diffusion coefficient in hydrated ionomer film.

        Returns
        -------
        float
            O2 diffusion coefficient [m2/s].
        """
        return (self.hydrated_o2_diffusion * 
                (water_content / 14) ** self.o2_diffusion_exponent *
                arrhenius_term(self.o2_diffusion_activation_energy, temperature, 353.15))

    def o2_permeability(self, water_content, temperature=353.15):
        """
        Estimate O2 permeability using volume fraction approach. 
        Data from Goshtasbi et al. (2020)

        Returns
        -------
        float
            O2 permeability [kmol/m/s/Pa].

        Reference
        ---------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        fv = self.water_vol_fraction(water_content, water_molar_volume(temperature))
        RT = ct.gas_constant * temperature
        return (6.74e-12 * np.exp(-21280e3/RT) + fv * 50.5e-12 * np.exp(-20470e3/RT)) * 1e-3

    def proton_conductivity(self, water_content, temperature):
        """
        Proton conductivity using empirical fits.

        Returns
        -------
        float
            Proton conductivity [S/m].
        """
        fv = self.water_vol_fraction(water_content, water_molar_volume(temperature))
        return self.conductivity_correction * 50 * (np.maximum(fv, 0.11) - 0.1) ** self.conductivity_exp * arrhenius_term(self.conductivity_activation_energy, temperature, 298.15)

    def hydroxide_conductivity(self, water_content, temperature):
        """
        Calculate the hydroxide conductivity of the ionomer based on water content and temperature.

        Parameters
        ----------
        water_content : float
            The water content of the membrane (not used in calculation but kept for consistency).
        temperature : float
            The temperature in Kelvin (K).

        Returns
        -------
        float
            The hydroxide conductivity of the membrane in Siemens per meter (S/m).
        
        """
        # No hydroxide conductivity 
        return 1e-6 

    def equilibrium_water_content(self, rh):
        """
        Equilibrium water content as function of RH from Jinnouchi et al. (2021), fig S3a.

        Returns
        -------
        float
            Water content [n.d.].

        Reference
        ---------
        Jinnouchi, R. et al. Nat. Commun. 12, 4956 (2021).
        """
        return 21.669 * rh ** 3 - 27.692 * rh ** 2 + 17.624 * rh + 0.688

NafionD2020 = PFSAIonomer(dry_density=2004., equivalent_weight=952.)

@dataclass
class PAPIonomer(CatalystLayerIonomer):
    """
    A class representing a PAP ionomer, extending the CatalystLayerIonomer class.
    This class includes properties and methods for calculating hydroxide conductivity.

    References
    ----------
    Eon Chae, J. et al. J. Ind. Eng. Chem. 133, 255–262 (2024)
    Luo, X. et al. J. Memb. Sci. 598, 117680 (2020)
    Khalid, H. et al. Membranes (Basel) 12, 989 (2022).

    Attributes
    ----------
    dry_density : float
        Density of the membrane in kg/m³. Default is 1220 kg/m³.
    equivalent_weight : float
        Equivalent weight of the membrane in kg/kmol. Default is 1000/2.35 kg/kmol.

    Methods
    -------
    hydroxide_conductivity(water_content, temperature)
        Calculate the hydroxide conductivity based on water content and temperature.
    """
    # Data from Luo et al. (2020), table 1. 
    dry_density: float = 1220.
    equivalent_weight: float = 1000/2.35

    def hydroxide_conductivity(self, water_content, temperature):
        """
        Calculate the hydroxide conductivity of the ionomer based on water content and temperature.

        Parameters
        ----------
        water_content : float
            The water content of the membrane (not used in calculation but kept for consistency).
        temperature : float
            The temperature in Kelvin (K).

        Returns
        -------
        float
            The hydroxide conductivity of the membrane in Siemens per meter (S/m).
        
        References
        ----------
        Luo, X. et al. J. Memb. Sci. 598, 117680 (2020)
        Khalid, H. et al. Membranes (Basel) 12, 989 (2022).
        """
        # Room-temperature conductivity for liquid-equilibrated from Luo et al. (2020) with
        # activation energy from Khalid et al. (2022) for PAP-20. Liquid-equilibrated.
        return 5.8 * arrhenius_term(activation_energy=22.5e6,
                                              temperature=temperature,
                                              reference_temperature=298.15)
    