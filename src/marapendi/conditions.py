"""
Operating conditions for :mod:`marapendi` cell models.

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
from .constants import GAS_CONSTANT
from .water import water_saturation_pressure 

@dataclass
class SideOperatingConditions:
    """Inlet/outlet operating conditions for one electrode side.

    Attributes
    ----------
    reactant : str
        Reactant species consumed on this side (``'o2'`` for the cathode,
        ``'h2'`` for the anode).
    outlet_pressure : float
        Outlet gas pressure (Pa).
    inlet_pressure : float, optional
        Inlet gas pressure (Pa). Defaults to ``outlet_pressure`` if not given.
    inlet_temperature : float
        Inlet gas temperature (K), used to evaluate the saturation pressure
        for the inlet relative humidity. Defaults to 353.15 K, matching
        :attr:`~marapendi.conditions.CellOperatingConditions.cell_temperature`'s
        default.
    relative_humidity : float
        Inlet relative humidity (0-1).
    dry_o2_mole_fraction : float
        O2 mole fraction in the dry inlet gas (0 on the anode side).
    dry_h2_mole_fraction : float
        H2 mole fraction in the dry inlet gas (0 on the cathode side).
    stoichiometry : float
        Stoichiometric ratio of supplied to consumed reactant. Ignored if
        ``inlet_dry_molar_flow_rate`` is given.
    inlet_dry_molar_flow_rate : float, optional
        Inlet *dry* gas molar flow rate (same molar unit as
        :meth:`~marapendi.gas.GasModel.concentration`, i.e. mol/m^3 *
        m^3/s). If given, this is used directly (converted to a wet
        volumetric flow rate at the inlet conditions) instead of deriving
        the inlet flow rate from ``stoichiometry``.
    """
    reactant: str = 'o2'
    outlet_pressure: float = 1e5
    inlet_pressure: float | None = None
    inlet_temperature: float = 353.15
    relative_humidity: float = 0.5
    dry_o2_mole_fraction: float = 0.
    dry_h2_mole_fraction: float = 0.
    stoichiometry: float = 2.
    inlet_dry_molar_flow_rate: float | None = None

    def __post_init__(self):
        if self.inlet_pressure is None:
            self.inlet_pressure = self.outlet_pressure
        self.inlet_saturation_pressure = water_saturation_pressure(self.inlet_temperature)
        self.inlet_vapor_pressure = self.inlet_saturation_pressure * self.relative_humidity
        self.inlet_vapor_mole_fraction = self.inlet_vapor_pressure / self.inlet_pressure
        dry_reactant_mole_fraction = {
            'o2': self.dry_o2_mole_fraction, 'h2': self.dry_h2_mole_fraction,
        }[self.reactant]
        self.reactant_mole_fraction = dry_reactant_mole_fraction * (1 - self.inlet_vapor_mole_fraction)

    @property
    def inlet_gas_concentration(self) -> float:
        """Total molar concentration of the inlet gas mixture, at inlet conditions."""
        return self.inlet_pressure / (GAS_CONSTANT * self.inlet_temperature)

    @property
    def cell_pressure(self) -> float:
        """Average of inlet and outlet pressure (Pa), used as the cell-side pressure."""
        return (self.inlet_pressure + self.outlet_pressure) / 2

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
        default_factory=lambda: SideOperatingConditions(reactant='o2', dry_o2_mole_fraction=0.21)
    )
    an: SideOperatingConditions = field(
        default_factory=lambda: SideOperatingConditions(reactant='h2', dry_h2_mole_fraction=1.0)
    )
