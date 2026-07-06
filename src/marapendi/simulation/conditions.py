"""Operating conditions for steady-state PEMFC simulations.

:class:`SideConditions` holds the steady-state inlet boundary conditions for
one side of the cell (cathode or anode).

:class:`CellConditions` bundles ``current_density``, ``cell_temperature``, and
one :class:`SideConditions` per side into a single object that is passed to
:meth:`~marapendi.cell.ExplicitSteadyStateModel.set_initial_conditions` and
:meth:`~marapendi.cell.ExplicitSteadyStateModel.solve`.

:class:`OperatingConditions` is a backward-compatible alias for
:class:`SideConditions`.  :class:`DynamicOperatingConditions` wraps
:class:`SideConditions` fields as callables of time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..electrolyte.electrolyte import ElectrolyteSolution


@dataclass
class SideConditions:
    """Inlet boundary conditions for one side of the cell (cathode or anode).

    Attributes
    ----------
    inlet_temperature : float
        Gas inlet temperature (K).
    inlet_pressure : float
        Inlet pressure (Pa).  Defaults to ``outlet_pressure`` when omitted.
    outlet_pressure : float
        Outlet pressure (Pa).  Defaults to ``inlet_pressure`` when omitted.
    dry_o2_mole_fraction : float
        O₂ mole fraction in the dry gas stream.
    dry_h2_mole_fraction : float
        H₂ mole fraction in the dry gas stream.
    inlet_relative_humidity : float
        Relative humidity at the inlet (0–1).
    stoichiometry : float
        Inlet stoichiometric ratio (actual / required flow).
    inlet_liquid_saturation : float
        Volume fraction of the liquid phase at the inlet.
    inlet_liquid : ElectrolyteSolution
        Liquid phase composition.
    inlet_liquid_flow_rate : float
        Liquid volumetric flow rate at the inlet (m³/s).
    inlet_gas_flow_rate : float
        Gas volumetric flow rate at the inlet (m³/s).
    """

    inlet_temperature: float = 353.15
    inlet_pressure: float = None
    outlet_pressure: float = None
    dry_o2_mole_fraction: float = 0.2
    dry_h2_mole_fraction: float = 0.0
    inlet_relative_humidity: float = 0.5
    stoichiometry: float = 2
    inlet_liquid_saturation: float = 0
    inlet_liquid: ElectrolyteSolution = field(default_factory=ElectrolyteSolution)
    inlet_liquid_flow_rate: float = 0
    inlet_gas_flow_rate: float = 0

    def __post_init__(self):
        if self.inlet_pressure is None:
            self.inlet_pressure = self.outlet_pressure if self.outlet_pressure is not None else 101325.0
        if self.outlet_pressure is None:
            self.outlet_pressure = self.inlet_pressure
        self.average_pressure = 0.5 * (self.inlet_pressure + self.outlet_pressure)


# Backward-compatible alias.
OperatingConditions = SideConditions


@dataclass
class CellConditions:
    """Full set of operating conditions for a steady-state cell simulation.

    Bundles the current density, stack temperature, and one
    :class:`SideConditions` per side into a single object for use with
    :meth:`~marapendi.cell.ExplicitSteadyStateModel.set_initial_conditions`
    and :meth:`~marapendi.cell.ExplicitSteadyStateModel.solve`.

    Attributes
    ----------
    current_density : float or ndarray
        Current density (A/m²).  Can be a scalar or a 1-D array to evaluate
        the full polarization curve in one vectorised call.
    cell_temperature : float or ndarray
        Stack operating temperature (K).  Must be broadcastable with
        ``current_density``.
    inlet_cooling_temperature : float or ndarray
        Stack inlet cooling liquid temperature (K).  Must be broadcastable with
        ``current_density``.
    outlet_cooling_temperature : float or ndarray
        Stack outlet cooling liquid temperature (K).  Must be broadcastable with
        ``current_density``.
    ca : SideConditions
        Cathode inlet conditions.
    an : SideConditions
        Anode inlet conditions.
    """

    current_density: float | np.ndarray
    cell_temperature: float | np.ndarray = None 
    inlet_cooling_temperature: float | np.ndarray = None 
    outlet_cooling_temperature: float | np.ndarray = None 
    ca: SideConditions = None 
    an: SideConditions = None 

    def __post_init__(self): 
        if self.cell_temperature is None: 
            self.cell_temperature = (self.inlet_cooling_temperature + self.outlet_cooling_temperature) / 2
        else: 
            self.outlet_cooling_temperature = self.cell_temperature 
            self.inlet_cooling_temperature = self.cell_temperature - 1

class DynamicOperatingConditions:
    """Time-varying inlet boundary conditions for one side of the cell.

    Each attribute is a callable ``f(t)`` that returns the value at time *t*.
    Call :meth:`get_operating_conditions` to snapshot a steady-state
    :class:`OperatingConditions` at a given time.
    """

    def __init__(
        self,
        inlet_temperature=lambda t: 353.15,
        inlet_pressure=None,
        outlet_pressure=None,
        dry_o2_mole_fraction=lambda t: 0.2,
        dry_h2_mole_fraction=lambda t: 0.0,
        inlet_relative_humidity=lambda t: 0.5,
        stoichiometry=lambda t: 2,
        inlet_liquid_saturation=lambda t: 0,
        inlet_liquid=ElectrolyteSolution(),
        inlet_liquid_flow_rate=lambda t: 0,
        inlet_gas_flow_rate=lambda t: 0,
    ):
        self.inlet_temperature = inlet_temperature
        self.inlet_pressure = inlet_pressure
        self.outlet_pressure = outlet_pressure
        self.dry_o2_mole_fraction = dry_o2_mole_fraction
        self.dry_h2_mole_fraction = dry_h2_mole_fraction
        self.inlet_relative_humidity = inlet_relative_humidity
        self.stoichiometry = stoichiometry
        self.inlet_liquid_saturation = inlet_liquid_saturation
        self.inlet_liquid = inlet_liquid
        self.inlet_liquid_flow_rate = inlet_liquid_flow_rate
        self.inlet_gas_flow_rate = inlet_gas_flow_rate

        if self.inlet_pressure is None:
            self.inlet_pressure = self.outlet_pressure if self.outlet_pressure is not None else lambda t: 101325.0
        if self.outlet_pressure is None:
            self.outlet_pressure = self.inlet_pressure

    def get_operating_conditions(self, t) -> OperatingConditions:
        """Return an :class:`OperatingConditions` snapshot at time *t*."""
        return OperatingConditions(
            self.inlet_temperature(t),
            self.inlet_pressure(t),
            self.outlet_pressure(t),
            self.dry_o2_mole_fraction(t),
            self.dry_h2_mole_fraction(t),
            self.inlet_relative_humidity(t),
            self.stoichiometry(t),
            self.inlet_liquid_saturation(t),
            self.inlet_liquid,
            self.inlet_liquid_flow_rate(t),
            self.inlet_gas_flow_rate(t),
        )
