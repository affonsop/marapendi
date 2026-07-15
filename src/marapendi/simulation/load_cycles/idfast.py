"""
ID-FAST load cycle for transient PEMFC simulations.

Implements the **Improved Dynamic Fuel-cell ASsessment Test** protocol as
described in Colombo et al., *J. Power Sources* **553**, 232–250 (2023).

The cycle consists of a 2005 s low-power cold section followed by a 1920 s
high-power hot section (total 3925 s).  During the high-power section the
cell temperature ramps linearly from the cold setpoint to the hot setpoint
between the start of the first C4 step and the end of the second C4 step,
then ramps back to the cold setpoint by t = 3725 s, after which it holds
constant until the end of the cycle (t = 3925 s).

Dew points switch from cold to hot at t = 2745 s (start of first C4) and
back to cold at t = 3725 s (end of temperature ramp-back).

Usage
-----
>>> from marapendi.simulation.load_cycles.idfast import IDFastCycle
>>> cycle = IDFastCycle(
...     current_densities={'C0': 950., 'C1': 2470., 'C2': 5890.,
...                        'C3': 12730., 'C4': 17480.}
... )
>>> # Use directly as the conditions callable for TransientModel:
>>> sol = tr_model.solve(cell, cycle, t_span=(0, cycle.duration))
"""
from __future__ import annotations

import numpy as np

from ..conditions import DynamicSideConditions
from .load_cycles import LoadCycle, PiecewiseProfile

# ── step tables ────────────────────────────────────────────────────────────────
# Each entry is (level_name, dwell_seconds).
# Levels: 'C0'–'C4' are power levels; '0' is the idle stop.

_LOW_STEPS: list[tuple[str, int]] = (
    [('C0', 120), ('C1', 50)] + 
    [('C0', 200), ('C1', 50), ('C2', 40), ('C1', 50)] * 5 + 
    [('C0', 100), ('0',  320)]
)
_HIGH_STEPS: list[tuple[str, int]] = (
    [('C0', 100), ('C1', 50), ('C2', 40), ('C1', 50),('C0', 200)] + 
    [('C2', 50), ('C3', 50)] * 3 + 
    [('C4', 220), ('C3', 50)] * 2 +
    [('C2', 40),('C0', 280), ('0',  320)]
)

_LOW_STARTS  = np.concatenate([[0], np.cumsum([d for _, d in _LOW_STEPS])])
_HIGH_STARTS = np.concatenate([[0], np.cumsum([d for _, d in _HIGH_STEPS])])

#: Duration of the low-power section (s).
LOW_DURATION: int  = int(_LOW_STARTS[-1])   # 2005 s
#: Duration of the high-power section (s).
HIGH_DURATION: int = int(_HIGH_STARTS[-1])  # 1920 s
#: Total cycle duration (s).
CYCLE_DURATION: int = LOW_DURATION + HIGH_DURATION  # 3925 s

# Temperature ramp anchor times within the high-power section (relative to section start).
_T_RAMP_START = float(_HIGH_STARTS[11])   # 740 s  — start of first  C4 step
_T_RAMP_PEAK  = float(_HIGH_STARTS[14])   # 1230 s — end   of second C4 step
_T_RAMP_BACK  = 1720.                     # 1720 s — temperature returns to T_cold

# Absolute times for temperature and dew-point transitions.
_T_ABS_RAMP_UP_START = LOW_DURATION + _T_RAMP_START  # 2745 s
_T_ABS_RAMP_UP_END   = LOW_DURATION + _T_RAMP_PEAK   # 3235 s
_T_ABS_RAMP_DOWN_END = LOW_DURATION + _T_RAMP_BACK   # 3725 s

# Default pressures (P_ca, P_an) in Pa for each level (Colombo 2023, Table 3).
_DEFAULT_PRESSURES: dict[str, tuple[float, float]] = {
    'C0': (1.40e5, 1.90e5),
    'C1': (1.40e5, 1.90e5),
    'C2': (1.40e5, 1.90e5),
    'C3': (2.48e5, 2.75e5),
    'C4': (2.80e5, 3.00e5),
    '0':  (1.01e5, 1.01e5),
}


def _steps_to_segs(
    steps: list[tuple[str, int]],
    level_values: dict,
    offset: float = 0.,
) -> list[tuple[str, float, float]]:
    """Convert a step table to PiecewiseProfile segment list.

    Parameters
    ----------
    steps : list of (level_name, dwell_s)
    level_values : dict mapping level_name → float
    offset : float
        Absolute time of the first step's start (s).
    """
    segs = []
    t = offset
    for level, dwell in steps:
        t += dwell
        segs.append(('const', float(level_values[level]), t))
    return segs


class IDFastCycle(LoadCycle):
    """ID-FAST load cycle for transient PEMFC simulation.

    Subclasses :class:`~marapendi.simulation.load_cycles.LoadCycle` and is
    callable as ``cycle(t) -> CellConditions``, so it can be passed directly
    to :meth:`~marapendi.models.base.transient.TransientModel.solve`.

    All time-varying channels are represented internally as
    :class:`~marapendi.simulation.load_cycles.PiecewiseProfile` objects.

    Parameters
    ----------
    current_densities : dict
        Mapping of level name to current density (A m⁻²).
        Required keys: ``'C0'``, ``'C1'``, ``'C2'``, ``'C3'``, ``'C4'``.
        The stop-level key ``'0'`` defaults to 100 A m⁻² if omitted.
    pressures : dict, optional
        Mapping of level name to ``(P_ca, P_an)`` in Pa.  Defaults to the
        values from Colombo 2023, Table 3 (see :data:`_DEFAULT_PRESSURES`).
    T_cold : float
        Cell temperature setpoint during the cold regime (°C).  Default 71.
    T_hot : float
        Peak cell temperature during the hot regime (°C).  Default 90.
    dT_cool : float
        Coolant temperature spread: outlet − inlet (K).  Default 4.
    dew_point_an_cold : float
        Anode dew-point temperature in the cold regime (°C).  Default 58.
    dew_point_ca_cold : float
        Cathode dew-point temperature in the cold regime (°C).  Default 43.
    dew_point_an_hot : float
        Anode dew-point temperature in the hot regime (°C).  Default 72.
    dew_point_ca_hot : float
        Cathode dew-point temperature in the hot regime (°C).  Default 57.
    stoichiometry_ca : float
        Cathode stoichiometric ratio.  Default 1.6.
    stoichiometry_an : float
        Anode stoichiometric ratio.  Default 1.4.
    dry_o2_mole_fraction : float
        Cathode dry O2 mole fraction during power-producing ('C0'-'C4') levels.  Default 0.21.
    dry_o2_mole_fraction_idle : float
        Cathode dry O2 mole fraction during the idle-stop ('0') level, when
        air flow is cut off.  Default 0.05.
    time_step : float
        Time-grid step for :attr:`~LoadCycle.cycle_time` (s).  Default 1.0.

    Examples
    --------
    >>> cycle = IDFastCycle(
    ...     current_densities={'C0': 950., 'C1': 2470., 'C2': 5890.,
    ...                        'C3': 12730., 'C4': 17480.}
    ... )
    >>> cycle.duration      # 3925 s
    >>> cycle.low_duration  # 2005 s
    >>> sol = tr_model.solve(cell, cycle, t_span=(0, cycle.duration))
    """

    def __init__(
        self,
        current_densities: dict[str, float] | None = None,
        pressures: dict[str, tuple[float, float]] | None = None,
        T_cold: float = 71.,
        T_hot: float  = 90.,
        dT_cool: float = 0,
        dew_point_an_cold: float = 58.,
        dew_point_ca_cold: float = 43.,
        dew_point_an_hot: float  = 72.,
        dew_point_ca_hot: float  = 57.,
        stoichiometry_ca: float = 1.6,
        stoichiometry_an: float = 1.4,
        cathode_minimum_current_for_stoich: float = 0.e4,
        anode_minimum_current_for_stoich: float = 0.e4,
        dry_o2_mole_fraction: float = 0.21,
        dry_o2_mole_fraction_idle: float = 0.00001,
        time_step: float = 1.0,
    ):
        if current_densities is None: 
            current_densities = {
                'C0': 950.,    # 0.095 A cm⁻²
                'C1': 2470.,   # 0.247 A cm⁻²
                'C2': 5890.,   # 0.589 A cm⁻²
                'C3': 12730.,  # 1.273 A cm⁻²
                'C4': 17480.,  # 1.748 A cm⁻²
            }
        required = {'C0', 'C1', 'C2', 'C3', 'C4'}
        missing = required - set(current_densities)
        if missing:
            raise ValueError(f"Missing current density levels: {sorted(missing)}")

        i_lvl = {**current_densities, '0': current_densities.get('0', 100.)}
        p_lvl = _DEFAULT_PRESSURES if pressures is None else pressures

        # ── current density profile ────────────────────────────────────────────
        i_profile = PiecewiseProfile(
            _steps_to_segs(_LOW_STEPS,  i_lvl, offset=0.)
            + _steps_to_segs(_HIGH_STEPS, i_lvl, offset=float(LOW_DURATION))
        )

        # ── pressure profiles ──────────────────────────────────────────────────
        p_ca_profile = PiecewiseProfile(
            _steps_to_segs(_LOW_STEPS,  {k: v[0] for k, v in p_lvl.items()}, 0.)
            + _steps_to_segs(_HIGH_STEPS, {k: v[0] for k, v in p_lvl.items()}, float(LOW_DURATION))
        )
        p_an_profile = PiecewiseProfile(
            _steps_to_segs(_LOW_STEPS,  {k: v[1] for k, v in p_lvl.items()}, 0.)
            + _steps_to_segs(_HIGH_STEPS, {k: v[1] for k, v in p_lvl.items()}, float(LOW_DURATION))
        )

        # ── temperature profile ────────────────────────────────────────────────
        T_cold_K = 273.15 + T_cold
        T_hot_K  = 273.15 + T_hot
        T_profile = PiecewiseProfile([
            ('const', T_cold_K, _T_ABS_RAMP_UP_START),  # [0, 2745]    hold cold
            ('ramp',  T_hot_K,  _T_ABS_RAMP_UP_END),    # [2745, 3235] ramp up
            ('ramp',  T_cold_K, _T_ABS_RAMP_DOWN_END),  # [3235, 3725] ramp down
            ('const', T_cold_K, float(CYCLE_DURATION)), # [3725, 3925] hold cold
        ])

        # ── dew-point profiles (K) ─────────────────────────────────────────────
        # Switch cold → hot at ramp-up start, hot → cold at ramp-down end.
        dew_ca_profile = PiecewiseProfile([
            ('const', 273.15 + dew_point_ca_cold, _T_ABS_RAMP_UP_START),
            ('const', 273.15 + dew_point_ca_hot,  _T_ABS_RAMP_DOWN_END-280),
            ('const', 273.15 + dew_point_ca_cold, float(CYCLE_DURATION)),
        ])
        dew_an_profile = PiecewiseProfile([
            ('const', 273.15 + dew_point_an_cold, _T_ABS_RAMP_UP_START),
            ('const', 273.15 + dew_point_an_hot,  _T_ABS_RAMP_DOWN_END-280),
            ('const', 273.15 + dew_point_an_cold, float(CYCLE_DURATION)),
        ])

        # ── cathode dry O2 mole fraction profile ───────────────────────────────
        # Drops during the idle-stop ('0') level, when air flow is cut off.
        o2_lvl = {**{k: dry_o2_mole_fraction for k in i_lvl}, '0': dry_o2_mole_fraction_idle}
        o2_profile = PiecewiseProfile(
            _steps_to_segs(_LOW_STEPS,  o2_lvl, offset=0.)
            + _steps_to_segs(_HIGH_STEPS, o2_lvl, offset=float(LOW_DURATION))
        )

        super().__init__(
            duration=CYCLE_DURATION,
            time_step=time_step,
            current_density=i_profile,
            cell_temperature=T_profile,
            dT_cool=dT_cool,
            ca=DynamicSideConditions(
                outlet_pressure=p_ca_profile,
                dew_point_temperature=dew_ca_profile,
                stoichiometry=stoichiometry_ca,
                dry_o2_mole_fraction=o2_profile,
                minimum_current_density_for_stoich=cathode_minimum_current_for_stoich
            ),
            an=DynamicSideConditions(
                outlet_pressure=p_an_profile,
                dew_point_temperature=dew_an_profile,
                stoichiometry=stoichiometry_an,
                dry_h2_mole_fraction=1.0,
                minimum_current_density_for_stoich=anode_minimum_current_for_stoich
            ),
        )

    # ── public properties ──────────────────────────────────────────────────────

    @property
    def low_duration(self) -> int:
        """Duration of the low-power section (s)."""
        return LOW_DURATION

    @property
    def high_duration(self) -> int:
        """Duration of the high-power section (s)."""
        return HIGH_DURATION
