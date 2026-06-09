"""
Operating conditions for dynamic PEM/AEM cell simulations.

Classes
-------
OperatingConditions
    Frozen per-side snapshot at one instant.
CellSnapshot
    Frozen cell-level snapshot (current_density + two OperatingConditions).
SideConditions
    Time-varying conditions for one electrode side; .at(t) → OperatingConditions.
CellConditions
    Full cell conditions; .at(t) → CellSnapshot.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np 
from marapendi.models.water import water_molecular_weight, water_saturation_pressure



@dataclass
class OperatingConditions:
    """Frozen per-side boundary conditions at a single instant."""
    temperature:                float
    backpressure:               float
    inlet_gas_molar_flow_rates: np.ndarray 
    inlet_h2ol_molar_flow_rate: float 

    def __post_init__(self): 
        self.inlet_gas_molar_flow_rate = np.sum(self.inlet_gas_molar_flow_rates, axis=0)
        self.inlet_liquid_mass_flow_rate = self.inlet_h2ol_molar_flow_rate * water_molecular_weight

@dataclass
class CellSnapshot:
    """All operating conditions sampled at a single instant *t*."""
    current_density: float
    temperature: float
    ca: OperatingConditions
    an: OperatingConditions


class SideConditions:
    """Time-varying conditions for one electrode side.

    Every field is a callable ``f(t: float) -> float`` or a plain float
    (treated as a constant).  Call :meth:`at` to obtain a frozen
    :class:`OperatingConditions` snapshot.

    Parameters
    ----------
    temperature : callable or float
        Boundary temperature [K] (channel wall / coolant).
    backpressure : callable or float
        Gas backpressure [Pa].
    inlet_h2ov_molar_flow_rate : callable or float
        Relative humidity at the inlet, 0–1.
    inlet_o2_molar_flow_rate : callable or float
        O₂ mole fraction in the dry gas mixture.
    inlet_h2_molar_flow_rate : callable or float
        H₂ mole fraction in the dry gas mixture.
    stoichiometry : callable or float
        Stoichiometric ratio of the supplied reactant to the consumed amount.
        Use a very large value (e.g. 1000) to model a differential cell where
        the gas composition is uniform (no depletion along the channel).
    liquid_saturation : callable or float
        Liquid water saturation at the channel/GDL boundary (s_C), 0–1.
    """

    def __init__(
        self,
        temperature:                  float | object = lambda t: 353.15,
        backpressure:                 float | object = lambda t: 1.5e5,
        inlet_gas_molar_flow_rates:   np.ndarray | object = lambda t: np.array([0,0,0,0]),
        inlet_h2ol_molar_flow_rate:   float | object = lambda t: 0.,
        channel_liquid_saturation:    float | object = 0.,
    ):
        def _wrap(v):
            return v if callable(v) else (lambda _t, _v=v: _v)

        self.temperature                = _wrap(temperature)
        self.backpressure               = _wrap(backpressure)
        self.inlet_gas_molar_flow_rates = _wrap(inlet_gas_molar_flow_rates)
        self.inlet_h2ol_molar_flow_rate = _wrap(inlet_h2ol_molar_flow_rate)
        self.channel_liquid_saturation  = _wrap(channel_liquid_saturation)

    def at(self, t: float | np.ndarray) -> OperatingConditions:
        """Return a frozen snapshot at time *t*."""
        return OperatingConditions(
            temperature                 = self.temperature(t),
            backpressure                = self.backpressure(t),
            inlet_gas_molar_flow_rates  = self.inlet_gas_molar_flow_rates(t),
            inlet_h2ol_molar_flow_rate  = self.inlet_h2ol_molar_flow_rate(t),
        )

class InletAirConditions(SideConditions):
    def __init__(
        self,
        temperature:                  float | object = lambda t: 353.15,
        backpressure:                 float | object = lambda t: 1.5e5,
        rh_ref_pressure:              float | object = lambda t: 1.5e5,
        o2_molar_flow_rate:           float | object = lambda t: 0.,
        o2_dry_mole_fraction:         float | object = lambda t: 0.21,
        inlet_rh:                     float | object = lambda t: 0.,
        inlet_h2ol_molar_flow_rate:   float | object = lambda t: 0.,
        channel_liquid_saturation:    float | object = 0.,
    ):
        def _wrap(v):
            return v if callable(v) else (lambda _t, _v=v: _v)

        self.temperature                = _wrap(temperature)
        self.backpressure               = _wrap(backpressure)
        self.rh_ref_pressure            = _wrap(rh_ref_pressure)
        self.o2_molar_flow_rate         = _wrap(o2_molar_flow_rate)
        self.o2_dry_mole_fraction       = _wrap(o2_dry_mole_fraction)
        self.inlet_rh                   = _wrap(inlet_rh)
        self.inlet_h2ol_molar_flow_rate = _wrap(inlet_h2ol_molar_flow_rate)
        self.channel_liquid_saturation  = _wrap(channel_liquid_saturation)
        
        self.n2_molar_flow_rate = lambda t: (
            self.o2_molar_flow_rate(t) 
            / self.o2_dry_mole_fraction(t) 
            * (1-self.o2_dry_mole_fraction(t))
        )
       
        self.h2ov_mole_fraction = lambda t: (
            self.inlet_rh(t) 
            * water_saturation_pressure(self.temperature(t))
            / self.rh_ref_pressure(t) 
        )

    def at(self, t: float | np.ndarray) -> OperatingConditions:
        o2_molar_flow_rate = self.o2_molar_flow_rate(t)
        h2ov_mole_fraction = self.h2ov_mole_fraction(t)
        h2ov_molar_flow_rate = (
            o2_molar_flow_rate 
            / self.o2_dry_mole_fraction(t) 
            / (1-h2ov_mole_fraction) 
            * h2ov_mole_fraction
        )
        return OperatingConditions(
            temperature                 = self.temperature(t),
            backpressure                = self.backpressure(t),
            inlet_gas_molar_flow_rates  = np.array(
                [
                    o2_molar_flow_rate,
                    self.n2_molar_flow_rate(t),
                    np.zeros_like(t), 
                    h2ov_molar_flow_rate
                ]
            ),
            inlet_h2ol_molar_flow_rate  = self.inlet_h2ol_molar_flow_rate(t),
        )

class InletHydrogenConditions(SideConditions):
    def __init__(
        self,
        temperature:                  float | object = lambda t: 353.15,
        backpressure:                 float | object = lambda t: 1.5e5,
        rh_ref_pressure:              float | object = lambda t: 1.5e5,
        h2_molar_flow_rate:           float | object = lambda t: 0.,
        inlet_rh:                     float | object = lambda t: 0.,
        inlet_h2ol_molar_flow_rate:   float | object = lambda t: 0.,
        channel_liquid_saturation:    float | object = 0.,
    ):
        def _wrap(v):
            return v if callable(v) else (lambda _t, _v=v: _v)

        self.temperature                = _wrap(temperature)
        self.backpressure               = _wrap(backpressure)
        self.rh_ref_pressure            = _wrap(rh_ref_pressure)
        self.h2_molar_flow_rate         = _wrap(h2_molar_flow_rate)
        self.inlet_rh                   = _wrap(inlet_rh)
        self.inlet_h2ol_molar_flow_rate = _wrap(inlet_h2ol_molar_flow_rate)
        self.channel_liquid_saturation  = _wrap(channel_liquid_saturation)
        
        self.h2ov_mole_fraction = lambda t: (
            self.inlet_rh(t) 
            * water_saturation_pressure(self.temperature(t))
            / self.rh_ref_pressure(t) 
        )

    def at(self, t: float | np.ndarray) -> OperatingConditions:
        h2_molar_flow_rate = self.h2_molar_flow_rate(t)
        h2ov_mole_fraction = self.h2ov_mole_fraction(t)
        h2ov_molar_flow_rate = (
            h2_molar_flow_rate 
            / (1-h2ov_mole_fraction) 
            * h2ov_mole_fraction
        )
        return OperatingConditions(
            temperature                 = self.temperature(t),
            backpressure                = self.backpressure(t),
            inlet_gas_molar_flow_rates  = np.array(
                [
                    np.zeros_like(t),
                    np.zeros_like(t),
                    h2_molar_flow_rate, 
                    h2ov_molar_flow_rate
                ]
            ),
            inlet_h2ol_molar_flow_rate = self.inlet_h2ol_molar_flow_rate(t)
        )

class CellConditions:
    """Full cell operating conditions: current density and per-side conditions.

    Parameters
    ----------
    current_density : callable or float
        Applied current density [A m⁻²].  Scalar for galvanostatic
        experiments (update in a sweep loop); callable for dynamic profiles.
        May also be assigned directly as a plain float after construction
        (used by polarization-curve sweep loops).
    ca : SideConditions
        Cathode-side conditions.
    an : SideConditions
        Anode-side conditions.
    temperature : float, optional
        Convenience shortcut: if provided AND *ca* / *an* are not given,
        sets both sides to this constant temperature.
    """

    def __init__(
        self,
        current_density: float | object = lambda t: 0.,
        ca: SideConditions = None,
        an: SideConditions = None,
        temperature: float = None,
    ):
        def _wrap(v):
            return v if callable(v) else (lambda _t, _v=v: _v)
        
        self.current_density = _wrap(current_density)
        self.temperature = _wrap(temperature)
        
        self.ca = ca 
        self.an = an

    def at(self, t: float) -> CellSnapshot:
        """Return a frozen :class:`CellSnapshot` at time *t*.

        ``current_density`` may be a plain float (set directly) or a
        callable ``f(t) -> float``.
        """
        _cd = self.current_density
        return CellSnapshot(
            current_density = _cd(t) if callable(_cd) else float(_cd),
            temperature = self.temperature(t),
            ca = self.ca.at(t),
            an = self.an.at(t),
        )
