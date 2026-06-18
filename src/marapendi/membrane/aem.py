"""Poly(aryl piperidinium) (PAP) ionomer material properties."""
from __future__ import annotations

from dataclasses import dataclass

from ..tools import arrhenius_term

from .ionomer import Ionomer


@dataclass
class PAPIonomer(Ionomer):
    """Poly(aryl piperidinium) (PAP) ionomer.

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

        References
        ----------
        Luo, X. et al. J. Memb. Sci. 598, 117680 (2020)
        Khalid, H. et al. Membranes (Basel) 12, 989 (2022).
        """
        return 5.8 * arrhenius_term(
            activation_energy=22.5e6, temperature=temperature, reference_temperature=298.15,
        )
