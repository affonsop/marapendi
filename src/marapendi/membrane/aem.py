"""Poly(aryl piperidinium) (PAP) ionomer and anion-exchange membrane classes."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from ..tools import arrhenius_term

from .ionomer_base import Ionomer
from .membrane_base import Membrane


@dataclass
class PAPIonomer(Ionomer):
    """Poly(aryl piperidinium) (PAP) ionomer.

    Provides default values for all :class:`~marapendi.ionomer.Ionomer` fields.
    The hydroxide conductivity correlation is specific to PAP85; proton
    conductivity is negligible (``reference_conductivity = 0``).

    References
    ----------
    Eon Chae, J. et al. J. Ind. Eng. Chem. 133, 255-262 (2024)
    Luo, X. et al. J. Memb. Sci. 598, 117680 (2020)
    Khalid, H. et al. Membranes (Basel) 12, 989 (2022).
    """

    dry_density: float = 1220.
    equivalent_weight: float = 1000 / 2.35
    vapor_equilibrium_polynomial: list = field(default_factory=lambda: [14.41, -14.81, 13.13, 0.0])
    reference_conductivity: float = 0.
    conductivity_correction: float = 1.0
    reference_water_diffusivity: float = 4e-10
    reference_water_absorption_coefficient: float = 1e-6
    conductivity_exp: float = 1.5
    conductivity_fv_threshold: float = 0.11
    hydrated_proton_conductivity: float = 0.
    conductivity_activation_energy: float = 22.5e6
    water_diffusivity_activation_energy: float = 20e6 * 2.37
    water_absorption_activation_energy: float = 20e6 * 2.37
    reference_conductivity_temperature: float = 298.15
    reference_water_absorption_temperature: float = 303.15
    reference_water_diffusivity_temperature: float = 303.15

    def hydroxide_conductivity(self, water_content: float, temperature: float) -> float:
        """Hydroxide conductivity (S/m).

        References
        ----------
        Luo, X. et al. J. Memb. Sci. 598, 117680 (2020)
        Khalid, H. et al. Membranes (Basel) 12, 989 (2022).
        """
        return 5.8 * arrhenius_term(
            activation_energy=22.5e6, temperature=temperature, reference_temperature=298.15,
        )


@dataclass
class AEM(Membrane):
    """Anion-exchange membrane base class.

    Composes a :class:`PAPIonomer` and overrides charge-transport to use
    hydroxide as the charge carrier.

    Attributes
    ----------
    ionomer : PAPIonomer
        PAP ionomer providing hydroxide conductivity and water-transport correlations.
    """

    ionomer: PAPIonomer = field(default_factory=PAPIonomer)

    def equilibrium_water_content(self, rh: float, temperature: float, s_relax=None) -> float:
        """Equilibrium water content from the PAP sorption isotherm (n.d.)."""
        rh = np.clip(rh, 0, 1)
        a = self.ionomer.vapor_equilibrium_polynomial
        return ((a[0] * rh + a[1]) * rh + a[2]) * rh + a[3]

    def proton_conductivity(self, water_content: float, temperature: float) -> float:
        """Proton conductivity — negligible in AEMs (returns 1 µS/m)."""
        return 1e-6

    def calculate_electroosmotic_drag_coefficient(self, temperature: float, water_content: float) -> float:
        """Electroosmotic drag coefficient for hydroxide transport (n.d.)."""
        return -water_content / 14


@dataclass
class PAP85(AEM):
    """PAP85 poly(aryl piperidinium) anion-exchange membrane.

    References
    ----------
    Eon Chae, J. et al. J. Ind. Eng. Chem. 133, 255–262 (2024)
    Luo, X. et al. J. Memb. Sci. 598, 117680 (2020)
    Khalid, H. et al. Membranes (Basel) 12, 989 (2022).
    """
    pass
