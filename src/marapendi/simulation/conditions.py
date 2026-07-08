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
    minimum_current_density_for_stoich : float
        Minimum current density for which stoichiometry is applied (A/m2). 
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
    minimum_current_density_for_stoich: float = 0 

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
    time : float or ndarray
        Time steps in case of ndarray current density (s). 
    """

    current_density: float | np.ndarray
    cell_temperature: float | np.ndarray = None 
    inlet_cooling_temperature: float | np.ndarray = None 
    outlet_cooling_temperature: float | np.ndarray = None 
    ca: SideConditions = None 
    an: SideConditions = None 
    time: float | np.ndarray = None 

    def __post_init__(self): 
        if self.cell_temperature is None: 
            self.cell_temperature = (self.inlet_cooling_temperature + self.outlet_cooling_temperature) / 2
        else: 
            self.outlet_cooling_temperature = self.cell_temperature 
            self.inlet_cooling_temperature = self.cell_temperature - 1

def _psat(T_K):
    """Water saturation pressure (kPa) via the Magnus formula at T_K (K)."""
    T_C = np.asarray(T_K, dtype=float) - 273.15
    return 0.6105 * np.exp(17.27 * T_C / (T_C + 237.3))


def _eval_field(field, t):
    """Evaluate a field: if callable returns field(t), 
    if scalar returns broadcast constant."""
    if callable(field):
        return field(t)
    return np.full(np.asarray(t, dtype=float).shape or (1,), float(field))


class DynamicSideConditions:
    """Time-varying inlet conditions for one electrode side of the cell.

    Mirrors :class:`SideConditions` with each attribute being a callable,
    a :class:`~marapendi.simulation.load_cycles.PiecewiseProfile`, or a
    constant scalar.  Call ``side(t)`` to snapshot a
    :class:`SideConditions` at time *t*.

    Parameters
    ----------
    inlet_temperature : callable | float, optional
        Gas inlet temperature (K).  When *None*, the parent
        :class:`~marapendi.simulation.load_cycles.LoadCycle`'s
        ``cell_temperature`` is used as the default.
    outlet_pressure : callable | float, optional
        Outlet pressure (Pa).
    dry_o2_mole_fraction : callable | float
        O₂ mole fraction in the dry stream.  Default 0.21.
    dry_h2_mole_fraction : callable | float
        H₂ mole fraction in the dry stream.  Default 0.0.
    inlet_relative_humidity : callable | float, optional
        Inlet relative humidity (0–1).  Mutually exclusive with
        *dew_point_temperature*.
    dew_point_temperature : callable | float, optional
        Dew-point temperature (K).  Converted to RH via
        ``psat(T_dew) / psat(T_gas)`` at evaluation time.
        Mutually exclusive with *inlet_relative_humidity*.
    stoichiometry : callable | float
        Stoichiometric ratio.  Default 2.
    """

    def __init__(
        self,
        inlet_temperature=None,
        inlet_pressure=None,
        outlet_pressure=None,
        dry_o2_mole_fraction=0.21,
        dry_h2_mole_fraction=0.0,
        inlet_relative_humidity=None,
        dew_point_temperature=None,
        stoichiometry=2.,
        inlet_liquid_saturation=0.,
        inlet_liquid_flow_rate=0.,
        inlet_gas_flow_rate=0.,
        minimum_current_density_for_stoich=0.,
    ):
        self.inlet_temperature        = inlet_temperature
        self.inlet_pressure           = inlet_pressure
        self.outlet_pressure          = outlet_pressure
        self.dry_o2_mole_fraction     = dry_o2_mole_fraction
        self.dry_h2_mole_fraction     = dry_h2_mole_fraction
        self.inlet_relative_humidity  = inlet_relative_humidity
        self.dew_point_temperature    = dew_point_temperature
        self.stoichiometry            = stoichiometry
        self.inlet_liquid_saturation  = inlet_liquid_saturation
        self.inlet_liquid_flow_rate   = inlet_liquid_flow_rate
        self.inlet_gas_flow_rate      = inlet_gas_flow_rate
        self.minimum_current_density_for_stoich = minimum_current_density_for_stoich

    def __call__(self, t, *, default_inlet_T=None) -> 'SideConditions':
        """Return a :class:`SideConditions` at time *t* (s).

        Parameters
        ----------
        t : float
        default_inlet_T : float, optional
            Fallback inlet temperature (K) used when :attr:`inlet_temperature`
            is *None* (typically the parent cycle's cell temperature).
        """
        def _g(field, default=None):
            if field is None:
                return default
            return np.atleast_1d(_eval_field(field, t))

        T_gas = _g(self.inlet_temperature, 353.15)
        p_out = _g(self.outlet_pressure, 101325.)
        p_in  = _g(self.inlet_pressure,  p_out)

        if self.inlet_relative_humidity is not None:
            rh = _g(self.inlet_relative_humidity)
        elif self.dew_point_temperature is not None:
            T_dew = _g(self.dew_point_temperature)
            rh = float(_psat(T_dew) / _psat(T_gas))
        else:
            rh = 0.

        return SideConditions(
            inlet_temperature=T_gas,
            inlet_pressure=p_in,
            outlet_pressure=p_out,
            dry_o2_mole_fraction=_g(self.dry_o2_mole_fraction, 0.21),
            dry_h2_mole_fraction=_g(self.dry_h2_mole_fraction, 0.0),
            inlet_relative_humidity=rh,
            stoichiometry=_g(self.stoichiometry, 2.),
            inlet_liquid_saturation=_g(self.inlet_liquid_saturation, 0.),
            minimum_current_density_for_stoich=_g(self.minimum_current_density_for_stoich, 0)
        )


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
