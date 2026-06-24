
"""PFSA ionomer (e.g. Nafion) material properties."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from ..tools import arrhenius_term
from ..thermo.constants import GAS_CONSTANT
from ..thermo.water import water_molar_volume

from .ionomer_base import Ionomer
from .membrane_base import Membrane

@dataclass
class PFSAIonomer(Ionomer):
    """PFSA ionomer (e.g. Nafion) with empirical fits for proton conductivity and O2 transport."""
    equivalent_weight: float = 952.
    dry_density: float = 2004.
    vapor_equilibrium_polynomial: list = field(default_factory=lambda: [36, -39.85, 17.18, 0.043])
    reference_conductivity: float = 50
    conductivity_correction: float = 1.0
    reference_water_diffusivity: float = 4.3e-10
    reference_water_absorption_coefficient: float = 1e-5   
    conductivity_exp: float = 1.5
    conductivity_fv_threshold: float = 0.11
    hydrated_proton_conductivity: float = 11
    conductivity_activation_energy: float = 11e6
    water_diffusivity_activation_energy: float = 20e6
    water_absorption_activation_energy: float = 20e6
    reference_conductivity_temperature: float = 353.15
    reference_water_absorption_temperature: float = 353.15
    reference_water_diffusivity_temperature: float = 353.15
    hydrated_o2_diffusion: float = 1.14698e-10 * 14 ** 0.708
    o2_diffusion_exponent: float = 0.708
    o2_diffusion_activation_energy: float = 24e6

    def o2_film_diffusion_coefficient(self, water_content: float, temperature: float = 353.15) -> float:
        """Effective O2 diffusion coefficient in the hydrated ionomer film (m^2/s)."""
        return (
            self.hydrated_o2_diffusion * (water_content / 14) ** self.o2_diffusion_exponent
            * arrhenius_term(self.o2_diffusion_activation_energy, temperature, 353.15)
        )

    def h2_permeability(self, water_content: float, temperature: float = 353.15) -> float:
        """H2 permeability (kmol/m/s/Pa) from a volume-fraction approach.

        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        fv = self.water_vol_fraction(water_content, water_molar_volume(temperature))
        RT = GAS_CONSTANT * temperature
        return (15.7e-15 * np.exp(-20280e3 / RT) + fv * 45e-15 * np.exp(-18930e3 / RT))
    
    def o2_permeability(self, water_content: float, temperature: float = 353.15) -> float:
        """O2 permeability (kmol/m/s/Pa) from a volume-fraction approach.

        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        fv = self.water_vol_fraction(water_content, water_molar_volume(temperature))
        RT = GAS_CONSTANT * temperature
        return (6.74e-15 * np.exp(-21280e3 / RT) + fv * 50.5e-15 * np.exp(-20470e3 / RT))

    def calculate_electroosmotic_drag_coefficient(self, temperature: float, water_content: float) -> float:
        """Electroosmotic drag coefficient (n.d.) for a given ``water_content``."""
        return (0.02 * temperature - 3.86) / 22.5 * water_content

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


    def liquid_equilibrium_water_content(self, temperature):
        """Equilibrium water content in contact with liquid water.

        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        return 9.22 + 0.181 * (temperature - 273.15)

    def vapor_equilibrium_water_content(self, rh: float, temperature) -> float:
        """Equilibrium water content as a function of relative humidity.

        References
        ----------
        
        """
        a = self.vapor_equilibrium_polynomial
        return  ((a[0] * rh + a[1]) * rh + a[2]) * rh + a[3]
    
    def vapor_equilibrium_water_content_derivative(self, rh, temperature):
        a = self.vapor_equilibrium_polynomial
        return (3 * a[0] * rh + 2 * a[1]) * rh + a[2]



@dataclass
class PFSA(Membrane):
    """Perfluorosulfonic-acid (PFSA, e.g. Nafion) membrane.

    Attributes
    ----------
    conductivity_correction : float
        Correction factor for the proton conductivity correlation
        (Vetter and Schumacher, 2020).
    conductivity_exp : float
        Exponent of the proton conductivity correlation.
    conductivity_activation_energy : float
        Activation energy for proton conductivity (J/kmol).
    phi : float
        Contribution of relaxation phenomena to the ionomer water uptake,
        according to Goshtasbi et al. (2019).

    References
    ----------
    Springer, T. E. et al. J. Electrochem. Soc. 138, 2334 (1991).
    Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
    """
    ionomer: PFSAIonomer = field(default_factory=PFSAIonomer)
    relaxation_time_constant: float = 0.067
    relaxation_time_activation_energy: float = 28e6
    uptake_relaxed_fraction_constant: float = 0.014
    phi: float = 0.15

    def equilibrium_water_content(self, rh, temperature, s_relax=None):
        """Equilibrium water content from the Springer et al. (1991) polynomial isotherm.

        References
        ----------
        Springer, T. E. et al. J. Electrochem. Soc. 138, 2334 (1991).
        Goshtasbi et al. J. Electrochem. Soc. 2019, 166 (7), F3154.
        """
        rh = np.clip(rh, 0, 1)
        lmbd_eq_relaxed = self.ionomer.vapor_equilibrium_water_content(rh, temperature)
        return ((1 - self.phi) * lmbd_eq_relaxed + s_relax) if s_relax is not None else lmbd_eq_relaxed

    def equilibrium_water_content_derivative(self, rh, temperature, s_relax=None):
        rh = np.clip(rh, 0, 1)
        d_lmbd_eq_relaxed = self.ionomer.vapor_equilibrium_water_content_derivative(rh, temperature)
        return ((1 - self.phi) * d_lmbd_eq_relaxed + s_relax) if s_relax is not None else d_lmbd_eq_relaxed


    def liquid_equilibrium_water_content(self, temperature):
        """Equilibrium water content in contact with liquid water — delegates to ionomer."""
        return self.ionomer.liquid_equilibrium_water_content(temperature)

    def proton_conductivity(self, water_content_profile, temperature):
        return 1 / np.mean(
            1 / (
                self.ionomer.charge_conductivity(water_content_profile, 
                                                 temperature, 'proton')
            ),
            axis=0,
        )

    def proton_resistance(self, state, water_saturation=0):
        """Through-plane proton resistance (Ω·m²).

        Weights liquid- and vapor-equilibrated conductivities by water saturation.
        """
        average_conductivity = self.proton_conductivity(state.water_content_profile, state.temperature)
        return self.dry_thickness / average_conductivity

NafionD2020 = PFSAIonomer(
    dry_density=2004., 
    equivalent_weight=952., 
    vapor_equilibrium_polynomial=[21.669, -27.692, 17.624, 0.688] # Jinnouchi, R. et al. Nat. Commun. 12, 4956 (2021).
)
