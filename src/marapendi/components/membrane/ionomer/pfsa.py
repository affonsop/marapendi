"""PFSA ionomer (e.g. Nafion) material properties."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from ....tools import arrhenius_term
from ....models.constants import GAS_CONSTANT
from ....models.water import water_molar_volume

from .ionomer import Ionomer


@dataclass
class PFSAIonomer(Ionomer):
    """PFSA ionomer (e.g. Nafion) with empirical fits for proton conductivity and O2 transport."""

    dry_density: float = 2004.
    equivalent_weight: float = 952.
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

        References
        ----------
        Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
        """
        fv = self.water_vol_fraction(water_content, water_molar_volume(temperature))
        RT = GAS_CONSTANT * temperature
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

        References
        ----------
        Jinnouchi, R. et al. Nat. Commun. 12, 4956 (2021).
        """
        return 21.669 * rh ** 3 - 27.692 * rh ** 2 + 17.624 * rh + 0.688

    def tortuosity(self, volume_fraction: float) -> float:
        return np.where(
            volume_fraction > 0.16,
            1,
            0.0845 * (np.maximum(0.1, volume_fraction) - 0.04) ** -1.17,
        )


NafionD2020 = PFSAIonomer(dry_density=2004., equivalent_weight=952.)
