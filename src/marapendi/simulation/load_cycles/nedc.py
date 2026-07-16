"""
FC-DLC / NEDC load cycle for transient PEMFC simulations.

Implements the **Fuel Cell Dynamic Load Cycle** (FC-DLC), derived from the
New European Driving Cycle (NEDC) and standardised by JRC/FCH-JU
(Tsotridis et al., EUR 27632 EN, 2015, Appendix F).

The cycle consists of 35 piecewise-constant steps covering 1181 s, including
four urban sub-cycles and one extra-urban sub-cycle.  All operating conditions
other than current density are constant throughout.

Usage
-----
>>> from marapendi.simulation.load_cycles.nedc import NEDCCycle
>>> cycle = NEDCCycle(i_max=30_000.)        # 3.0 A cm⁻²
>>> cycle.duration                           # 1181 s
>>> state = tr_model.solve(cell, cycle, t_span=(0, cycle.duration))
"""
from __future__ import annotations

from ..conditions import DynamicSideConditions
from .load_cycles import LoadCycle, PiecewiseProfile

# ── FC-DLC step table ──────────────────────────────────────────────────────────
# Columns: [start_time_s, dwell_s, load_%]
# Source: Tsotridis et al., EUR 27632 EN (2015), Appendix F, Table F.1.
_FCDLC_TABLE: list[tuple[int, int, float]] = [
    (0,    15, 0.0  ),
    (15,   13, 12.5 ),
    (28,   33, 5.0  ),
    (61,   35, 26.7 ),
    (96,   47, 5.0  ),
    (143,  20, 41.7 ),
    (163,  25, 29.2 ),
    (188,  22, 5.0  ),
    (210,  13, 12.5 ),
    (223,  33, 5.0  ),
    (256,  35, 26.7 ),
    (291,  47, 5.0  ),
    (338,  20, 41.7 ),
    (358,  25, 29.2 ),
    (383,  22, 5.0  ),
    (405,  13, 12.5 ),
    (418,  33, 5.0  ),
    (451,  35, 26.7 ),
    (486,  47, 5.0  ),
    (533,  20, 41.7 ),
    (553,  25, 29.2 ),
    (578,  22, 5.0  ),
    (600,  13, 12.5 ),
    (613,  33, 5.0  ),
    (646,  35, 26.7 ),
    (681,  47, 5.0  ),
    (728,  20, 41.7 ),
    (748,  25, 29.2 ),
    (773,  68, 5.0  ),
    (841,  58, 58.3 ),
    (899,  82, 41.7 ),
    (981,  85, 58.3 ),
    (1066, 50, 83.3 ),
    (1116, 44, 100.0),
    (1160, 21, 0.0  ),
]

#: Total FC-DLC cycle duration (s).
NEDC_DURATION: int = 1181


class NEDCCycle(LoadCycle):
    """FC-DLC (NEDC-derived) load cycle for transient PEMFC simulation.

    Subclasses :class:`~marapendi.simulation.load_cycles.LoadCycle` and is
    callable as ``cycle(t) -> CellConditions``.  The 35 FC-DLC steps define a
    piecewise-constant current-density profile; all other operating conditions
    are constant throughout.

    Parameters
    ----------
    max_current_density : float
        Current density at 100 % load (A m⁻²).  Typical value: 17 000 A m⁻²
        (= 1.7 A cm⁻²).
    min_current_density : float
        Minimum operating current density (A m⁻²) — substituted for 0 % load
        steps to avoid numerical singularities.  Default 100 A m⁻².
    cell_temperature : float
        Isothermal cell temperature (K).  Default 343.15 (≈ 70 °C).
    cooling_temperature_increase : float
        Coolant temperature spread: outlet − inlet (K).  Default 0.
    cathode_inlet_pressure : float
        Cathode inlet pressure (Pa).  Default 2.30e5.
    anode_inlet_pressure : float
        Anode inlet pressure (Pa).  Default 2.50e5.
    cathode_outlet_pressure : float, optional
        Cathode outlet pressure (Pa).  Defaults to *cathode_inlet_pressure*
        when omitted.
    anode_outlet_pressure : float, optional
        Anode outlet pressure (Pa).  Defaults to *anode_inlet_pressure* when
        omitted.
    cathode_inlet_temperature : float
        Cathode inlet gas temperature (K).  Default 358.15 (≈ 85 °C).
    cathode_inlet_rh : float
        Cathode inlet relative humidity (–).  Default 0.3.
    anode_inlet_temperature : float
        Anode inlet gas temperature (K).  Default 358.15 (≈ 85 °C).
    anode_inlet_rh : float
        Anode inlet relative humidity (–).  Default 0.5.
    cathode_stoichiometry : float
        Cathode stoichiometric ratio.  Default 1.5.
    anode_stoichiometry : float
        Anode stoichiometric ratio.  Default 1.3.
    cathode_minimum_current_density_for_stoich : float
        Minimum current density below which cathode stoichiometry clamps to the
        value at this threshold (A m⁻²).  Default 2000.
    anode_minimum_current_density_for_stoich : float
        Minimum current density below which anode stoichiometry clamps to the
        value at this threshold (A m⁻²).  Default 2000.
    time_step : float
        Time-grid step for :attr:`~LoadCycle.cycle_time` (s).  Default 1.0.

    Examples
    --------
    >>> cycle = NEDCCycle(max_current_density=17_000.)
    >>> cycle.duration    # 1181 s
    >>> cycle.plot()
    """

    def __init__(
        self,
        max_current_density: float,
        *,
        min_current_density: float = 100.,
        cell_temperature: float = 353.15,
        cooling_temperature_increase: float = 0.,
        cathode_inlet_pressure: float = 2.30e5,
        anode_inlet_pressure: float = 2.50e5,
        cathode_outlet_pressure: float = None,
        anode_outlet_pressure: float = None,
        cathode_inlet_temperature: float = 358.15,
        cathode_inlet_rh: float = 0.3,
        anode_inlet_temperature: float = 358.15,
        anode_inlet_rh: float = 0.5,
        cathode_stoichiometry: float = 1.5,
        anode_stoichiometry: float = 1.3,
        cathode_minimum_current_density_for_stoich: float = 0.2e4, 
        anode_minimum_current_density_for_stoich: float = 0.2e4, 
        time_step: float = 1.0,
    ):
   

        i_segs = [
            ('const', max(pct / 100. * max_current_density, min_current_density), start + dwell)
            for start, dwell, pct in _FCDLC_TABLE
        ]

        super().__init__(
            duration=NEDC_DURATION,
            time_step=time_step,
            current_density=PiecewiseProfile(i_segs),
            cell_temperature=cell_temperature,
            dT_cool=cooling_temperature_increase,
            ca=DynamicSideConditions(
                inlet_pressure=cathode_inlet_pressure,
                outlet_pressure=cathode_outlet_pressure,
                inlet_temperature=cathode_inlet_temperature,
                inlet_relative_humidity=cathode_inlet_rh,
                stoichiometry=cathode_stoichiometry,
                dry_o2_mole_fraction=0.21,
                minimum_current_density_for_stoich=cathode_minimum_current_density_for_stoich, 
            ),
            an=DynamicSideConditions(
                inlet_pressure=anode_inlet_pressure,
                outlet_pressure=anode_outlet_pressure,
                inlet_temperature=anode_inlet_temperature,
                inlet_relative_humidity=anode_inlet_rh,
                stoichiometry=anode_stoichiometry,
                dry_h2_mole_fraction=1.0,
                minimum_current_density_for_stoich=anode_minimum_current_density_for_stoich,
            ),
        )
