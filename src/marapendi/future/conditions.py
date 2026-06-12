"""
Operating conditions for :mod:`marapendi.future` cell models.

Classes
-------
SideOperatingConditions
    Inlet operating conditions for one electrode side (cathode or anode).
CellOperatingConditions
    Full cell operating conditions: applied current density, cell
    temperature, and per-side :class:`SideOperatingConditions`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SideOperatingConditions:
    """Inlet operating conditions for one electrode side.

    Attributes
    ----------
    pressure : float
        Inlet gas pressure (Pa).
    relative_humidity : float
        Inlet relative humidity (0-1).
    dry_o2_mole_fraction : float
        O2 mole fraction in the dry inlet gas (0 on the anode side).
    dry_h2_mole_fraction : float
        H2 mole fraction in the dry inlet gas (0 on the cathode side).
    stoichiometry : float
        Stoichiometric ratio of supplied to consumed reactant.
    """

    pressure: float = 1e5
    relative_humidity: float = 0.5
    dry_o2_mole_fraction: float = 0.
    dry_h2_mole_fraction: float = 0.
    stoichiometry: float = 2.


@dataclass
class CellOperatingConditions:
    """Operating conditions for a steady-state cell-model evaluation.

    Attributes
    ----------
    current_density : float or np.ndarray
        Applied current density (A/m^2). May be an array for a polarization
        curve sweep.
    cell_temperature : float
        Cell (coolant/inlet) temperature (K).
    ca, an : SideOperatingConditions
        Cathode/anode inlet operating conditions.
    """

    current_density: float | np.ndarray
    cell_temperature: float = 353.15
    ca: SideOperatingConditions = field(
        default_factory=lambda: SideOperatingConditions(dry_o2_mole_fraction=0.21)
    )
    an: SideOperatingConditions = field(
        default_factory=lambda: SideOperatingConditions(dry_h2_mole_fraction=1.0)
    )
