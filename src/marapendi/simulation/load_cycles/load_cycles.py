"""
Time-varying load cycle definition for transient PEMFC simulations.

:class:`LoadCycle` mirrors the structure of
:class:`~marapendi.simulation.conditions.CellConditions`: every attribute
is a callable (or constant) of time, and calling ``cycle(t)`` returns a
:class:`CellConditions` snapshot. It is the natural time-variant counterpart
to the instantaneous :class:`CellConditions`.
"""
import numpy as np

from ..conditions import (
    CellConditions,
    DynamicSideConditions, _eval_field, _psat,
)


# ── small utilities ────────────────────────────────────────────────────────────

def _dewpoint_C(rh, T_K):
    """Dew-point temperature (°C) from relative humidity and gas temperature (K).

    Inverts the Magnus formula: RH = p_sat(T_dew) / p_sat(T_gas).
    """
    T_C = np.asarray(T_K) - 273.15
    gamma = 17.27 * T_C / (T_C + 237.3) + np.log(np.clip(rh, 1e-9, 1.0))
    return 237.3 * gamma / (17.27 - gamma)


def _veval(field, t):
    """Evaluate *field* at array *t* → 1-D ndarray (or None if field is None)."""
    if field is None:
        return None
    return np.atleast_1d(_eval_field(field, t))


# ── PiecewiseProfile ───────────────────────────────────────────────────────────

class PiecewiseProfile:
    """Piecewise constant or linear profile, callable as ``f(t) → np.ndarray``.

    Each segment is ``(kind, value, t_end)`` where

    * ``kind='const'`` — holds *value* for the entire segment.
    * ``kind='ramp'``  — linearly interpolates from the previous segment's end
      value to *value* by *t_end*.

    The first segment starts at t = 0.  Evaluation outside [0, last t_end]
    clamps to the first / last segment's value.

    Parameters
    ----------
    segments : list of (str, float | str, float)
        ``[(kind, value, t_end), ...]`` in chronological order.
    levels : dict, optional
        Maps string values to floats (resolves level names).

    Examples
    --------
    >>> prof = PiecewiseProfile([
    ...     ('const', 71., 2745.),   # hold 71 °C for 2745 s
    ...     ('ramp',  90., 3235.),   # ramp 71→90 °C over next 490 s
    ...     ('ramp',  71., 3725.),   # ramp 90→71 °C over next 490 s
    ...     ('const', 71., 3925.),   # hold 71 °C for final 200 s
    ... ])
    >>> prof(2745.)   # 71.0
    >>> prof(2990.)   # midpoint of up-ramp ≈ 80.5
    """

    def __init__(self, segments, levels=None):
        self._levels = levels or {}
        self._t_ends = np.array([t for _, _, t in segments], dtype=float)
        self._data: list[tuple[float, float, float, float]] = []
        t_start = 0.
        prev_end: float | None = None
        for kind, raw_val, t_end in segments:
            val = float(self._levels[raw_val]) if isinstance(raw_val, str) else float(raw_val)
            if kind == 'const':
                v0 = v1 = val
            else:  # 'ramp'
                v0 = prev_end if prev_end is not None else val
                v1 = val
            self._data.append((t_start, t_end, v0, v1))
            prev_end = v1
            t_start = t_end

        # Precomputed, per-segment arrays for vectorised evaluation (avoids a
        # Python loop over segments on every call — this is on the hot path
        # of the transient ODE right-hand side).
        data = np.asarray(self._data, dtype=float)
        self._seg_t0, self._seg_t1, self._seg_v0, self._seg_v1 = data.T
        self._seg_dt = self._seg_t1 - self._seg_t0

    def __call__(self, t):
        t_arr = np.asarray(t, dtype=float)
        scalar = t_arr.ndim == 0
        t_arr = np.atleast_1d(t_arr)
        # side='right': boundary point t==t_end belongs to the next segment.
        idx = np.clip(
            np.searchsorted(self._t_ends, t_arr, side='right'),
            0, len(self._data) - 1,
        )
        t0, v0, v1, dt = (
            self._seg_t0[idx], self._seg_v0[idx], self._seg_v1[idx], self._seg_dt[idx],
        )
        frac = np.where(dt > 0., np.clip((t_arr - t0) / dt, 0., 1.), 1.)
        result = v0 + (v1 - v0) * frac
        return float(result[0]) if scalar else result


# ── CycleSegment ───────────────────────────────────────────────────────────────

class CycleSegment:
    """One time segment in a load cycle, defining operating variables for a fixed dwell.

    Segments can be **summed** to build a :class:`LoadCycle` via
    :meth:`LoadCycle.from_segments`.

    Parameters
    ----------
    dwell : float
        Segment duration (s).  Mutually exclusive with *start* + *end*.
    start, end : float
        Alternative to *dwell*: segment spans ``[start, end]``.
    levels : dict, optional
        Mapping from string level names to float values.
    variables : dict, optional
        ``{channel_name: (kind, value)}`` — channel names use the
        ``'ca-outlet-pressure'`` hyphenated convention.
    **kwargs
        Shorthand: underscores in key names are mapped to hyphens,
        e.g. ``current_density=('const', 950.)`` → ``'current-density'``.

    Examples
    --------
    >>> seg_idle = CycleSegment(dwell=100, current_density=('const', 100.))
    >>> seg_peak = CycleSegment(dwell=50,  current_density=('const', 10_000.))
    >>> cycle = LoadCycle.from_segments([seg_idle, seg_peak])
    >>> cycle = seg_idle + seg_peak          # same result
    >>> cycle = sum([seg_idle, seg_peak, seg_idle])
    """

    def __init__(
        self,
        dwell=None,
        *,
        start=None,
        end=None,
        levels=None,
        variables=None,
        **kwargs,
    ):
        if dwell is not None:
            self.dwell = float(dwell)
        elif start is not None and end is not None:
            self.dwell = float(end) - float(start)
        else:
            raise ValueError("Specify dwell or (start and end).")

        self.levels = levels or {}
        self.variables: dict[str, tuple[str, object]] = {}
        if variables is not None:
            for k, v in variables.items():
                self.variables[k] = v if isinstance(v, tuple) else ('const', v)
        for k, v in kwargs.items():
            key = k.replace('_', '-')
            self.variables[key] = v if isinstance(v, tuple) else ('const', v)

    def __add__(self, other):
        if isinstance(other, CycleSegment):
            return LoadCycle.from_segments([self, other])
        if isinstance(other, LoadCycle) and hasattr(other, '_segments'):
            return LoadCycle.from_segments([self] + other._segments)
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        if isinstance(other, CycleSegment):
            return LoadCycle.from_segments([other, self])
        if isinstance(other, LoadCycle) and hasattr(other, '_segments'):
            return LoadCycle.from_segments(other._segments + [self])
        return NotImplemented


# ── LoadCycle ─────────────────────────────────────────────────────────────────

# Maps hyphenated flat-dict keys to DynamicSideConditions constructor kwargs.
_CA_KEY_MAP: dict[str, str] = {
    'ca-inlet-temperature':      'inlet_temperature',
    'ca-inlet-pressure':         'inlet_pressure',
    'ca-outlet-pressure':        'outlet_pressure',
    'ca-inlet-rh':               'inlet_relative_humidity',
    'ca-dew-point-temperature':  'dew_point_temperature',
    'ca-stoichiometry':          'stoichiometry',
    'ca-dry-o2-mole-fraction':   'dry_o2_mole_fraction',
    'ca-dry-h2-mole-fraction':   'dry_h2_mole_fraction',
}
_AN_KEY_MAP: dict[str, str] = {k.replace('ca-', 'an-'): v for k, v in _CA_KEY_MAP.items()}


class LoadCycle:
    """Time-varying load cycle for transient PEMFC simulations.

    Mirrors :class:`~marapendi.simulation.conditions.CellConditions`:
    every attribute is a callable, a
    :class:`PiecewiseProfile`, or a constant scalar; calling
    ``cycle(t)`` returns a :class:`CellConditions` snapshot, making
    any :class:`LoadCycle` directly usable as the *conditions* argument
    of :meth:`~marapendi.models.base.transient.TransientModel.solve`.

    Parameters
    ----------
    duration : float
        Total duration of one cycle (s).
    time_step : float
        Time-grid resolution for :attr:`cycle_time` (s).  Default 1.
    current_density : callable | float, optional
        Current density (A m⁻²).
    cell_temperature : callable | float, optional
        Cell / gas inlet temperature (K).  When given together with
        *dT_cool* and no explicit cooling temperatures, the coolant
        inlet and outlet temperatures are derived automatically.
    dT_cool : float
        Coolant temperature spread outlet − inlet (K).  Default 0.
    inlet_cooling_temperature : callable | float, optional
        Overrides the derived coolant inlet temperature.
    outlet_cooling_temperature : callable | float, optional
        Overrides the derived coolant outlet temperature.
    ca : DynamicSideConditions, optional
        Time-varying cathode inlet conditions.
    an : DynamicSideConditions, optional
        Time-varying anode inlet conditions.

    Examples
    --------
    Build a two-step load cycle directly:

    >>> from marapendi.simulation.load_cycles import (
    ...     LoadCycle, PiecewiseProfile, CycleSegment)
    >>> from marapendi.simulation.conditions import DynamicSideConditions
    >>> i_prof = PiecewiseProfile([('const', 1000., 100.), ('const', 10000., 150.)])
    >>> cycle = LoadCycle(
    ...     duration=150.,
    ...     current_density=i_prof,
    ...     cell_temperature=344.15,
    ...     dT_cool=4.,
    ...     ca=DynamicSideConditions(outlet_pressure=1.4e5, inlet_relative_humidity=0.265,
    ...                              stoichiometry=1.6, dry_o2_mole_fraction=0.21),
    ...     an=DynamicSideConditions(outlet_pressure=1.9e5, inlet_relative_humidity=0.558,
    ...                              stoichiometry=1.4, dry_h2_mole_fraction=1.0),
    ... )
    >>> cond = cycle(75.)   # CellConditions at t = 75 s
    """

    def __init__(
        self,
        duration: float,
        time_step: float = 1.0,
        *,
        current_density=None,
        cell_temperature=None,
        dT_cool: float = 0.,
        inlet_cooling_temperature=None,
        outlet_cooling_temperature=None,
        ca: DynamicSideConditions | None = None,
        an: DynamicSideConditions | None = None,
    ):
        self.duration   = float(duration)
        self.time_step  = float(time_step)
        self.cycle_time = np.linspace(0., self.duration,
                                      int(self.duration // self.time_step + 1))

        self.current_density = current_density
        self.cell_temperature = cell_temperature
        self.ca = ca
        self.an = an

        if inlet_cooling_temperature is None and cell_temperature is not None:
            T_f = cell_temperature
            dT_half = dT_cool / 2.
            self.inlet_cooling_temperature  = lambda t, _T=T_f, _d=dT_half: _eval_field(_T, t) - _d
            self.outlet_cooling_temperature = lambda t, _T=T_f, _d=dT_half: _eval_field(_T, t) + _d
        else:
            self.inlet_cooling_temperature  = inlet_cooling_temperature
            self.outlet_cooling_temperature = outlet_cooling_temperature

    # ── callable interface ─────────────────────────────────────────────────────

    def __call__(self, t: float = None, n_cycles=None) -> CellConditions:
        return self.conditions(t, n_cycles)

    def conditions(self, t: float = None, n_cycles=None) -> CellConditions:
        """Return :class:`CellConditions` at time *t* (s), periodic over :attr:`duration`."""
        
        if n_cycles: 
            t = np.arange(0, n_cycles * self.duration, self.time_step)
        else: 
            t = np.asarray(t, dtype=float)
        t_m = np.mod(t, self.duration)
        
        T_cell = np.atleast_1d(_eval_field(self.cell_temperature, t_m))
        
        T_in  = np.atleast_1d(_eval_field(self.inlet_cooling_temperature, t_m))
        T_out = np.atleast_1d(_eval_field(self.outlet_cooling_temperature, t_m))

        ca_cond = self.ca(t_m, default_inlet_T=T_cell) if self.ca is not None else None
        an_cond = self.an(t_m, default_inlet_T=T_cell) if self.an is not None else None
        i_val = np.atleast_1d(_eval_field(self.current_density, t_m))

        return CellConditions(
            current_density=np.atleast_1d(i_val),
            cell_temperature=T_cell,
            inlet_cooling_temperature=T_in,
            outlet_cooling_temperature=T_out,
            ca=ca_cond,
            an=an_cond,
            time=t, 
        )

    # ── integration hints ──────────────────────────────────────────────────────

    def discontinuity_times(self) -> np.ndarray:
        """Interior times (0 < t < duration) where any field's derivative is discontinuous.

        Collected from every :class:`PiecewiseProfile` field of this cycle
        (including nested ``ca``/``an`` side conditions). An ODE solver
        integrating through one of these kinks can't rely on its local error
        estimate — the smoothness assumption behind it is violated right at
        the kink — so :meth:`~marapendi.models.base.transient.TransientModel.solve`
        uses these times to split the integration into per-segment pieces.
        """
        times: set[float] = set()

        def _scan(obj):
            if obj is None:
                return
            for value in vars(obj).values():
                if isinstance(value, PiecewiseProfile):
                    times.update(float(t) for t in value._t_ends)
                elif isinstance(value, DynamicSideConditions):
                    _scan(value)

        _scan(self)
        interior = np.array(sorted(t for t in times if 0. < t < self.duration))
        # Round to merge near-duplicate breakpoints computed independently by different profiles.
        return np.unique(np.round(interior, 6))

    # ── vectorised evaluation ──────────────────────────────────────────────────

    def get_input_vectors(self, t=None, n_cycles=None) -> dict:
        """Evaluate all fields at array *t* and return a flat ``{key: ndarray}`` dict.

        Keys follow the ``'ca-inlet-temperature'`` hyphenated convention used
        by :meth:`plot`.  Only keys for which data exist are included.
        """
        if n_cycles: 
            t = np.arange(0, n_cycles * self.duration, 1.0)
        else: 
            t = np.asarray(t, dtype=float)
        v: dict = {}
        v['time'] = t 

        if self.current_density is not None:
            v['current-density'] = _veval(self.current_density, t)

        T_cell_arr = _veval(self.cell_temperature, t)
        if T_cell_arr is not None:
            v['cell-temperature'] = T_cell_arr

        if self.inlet_cooling_temperature is not None:
            v['inlet-cooling-temperature']  = _veval(self.inlet_cooling_temperature, t)
            v['outlet-cooling-temperature'] = _veval(self.outlet_cooling_temperature, t)

        for side, prefix in ((self.ca, 'ca'), (self.an, 'an')):
            if side is None:
                continue
            T_gas = (_veval(side.inlet_temperature, t)
                     if side.inlet_temperature is not None else T_cell_arr)
            if T_gas is not None:
                v[f'{prefix}-inlet-temperature'] = T_gas
            if side.outlet_pressure is not None:
                v[f'{prefix}-outlet-pressure'] = _veval(side.outlet_pressure, t)
            if side.inlet_relative_humidity is not None:
                v[f'{prefix}-inlet-rh'] = _veval(side.inlet_relative_humidity, t)
            elif side.dew_point_temperature is not None and T_gas is not None:
                T_dew = _veval(side.dew_point_temperature, t)
                v[f'{prefix}-dew-point-temperature'] = T_dew
                v[f'{prefix}-inlet-rh'] = _psat(T_dew) / _psat(T_gas)
            if side.stoichiometry is not None:
                v[f'{prefix}-stoichiometry'] = _veval(side.stoichiometry, t)
            if side.dry_o2_mole_fraction is not None:
                v[f'{prefix}-dry-o2-mole-fraction'] = _veval(side.dry_o2_mole_fraction, t)

        return v


    # ── plotting ───────────────────────────────────────────────────────────────

    def plot(self, t=None, figsize=None):
        """Plot operating condition time series for one cycle.

        Creates a multi-row figure with one subplot per condition category.
        Panels are shown only when the relevant data exists.

        Panels (in order):

        * **Current density** — ``'current-density'`` (A m⁻² → A cm⁻²)
        * **Pressures** — ``'ca-outlet-pressure'``, ``'an-outlet-pressure'`` (Pa → bar)
        * **Temperatures** — gas + coolant temperatures (K → °C)
        * **RH / dew points** — RH (%) on left axis, dew-point (°C) on right
        * **Stoichiometries**

        Parameters
        ----------
        t : array-like, optional
            Time vector (s).  Defaults to :attr:`cycle_time`.
        figsize : tuple, optional
            Figure size ``(width, height)`` in inches.

        Returns
        -------
        fig, axes
        """
        import matplotlib.pyplot as plt

        if t is None:
            t = self.cycle_time
        t = np.asarray(t, dtype=float)
        t_min = t / 60.0

        v = self.get_input_vectors(t)

        def _has(*keys):
            return any(v.get(k) is not None for k in keys)

        panels = []
        if _has('current-density'):
            panels.append('current')
        if _has('ca-outlet-pressure', 'an-outlet-pressure'):
            panels.append('pressure')
        if _has('ca-inlet-temperature', 'an-inlet-temperature',
                'inlet-cooling-temperature', 'outlet-cooling-temperature',
                'cell-temperature'):
            panels.append('temperature')
        if _has('ca-inlet-rh', 'an-inlet-rh'):
            panels.append('rh')
        if _has('ca-stoichiometry', 'an-stoichiometry'):
            panels.append('stoichiometry')

        if not panels:
            raise RuntimeError("No plottable data found.")

        n = len(panels)
        if figsize is None:
            figsize = (10, 2.4 * n)

        fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
        if n == 1:
            axes = [axes]

        panel_idx = 0

        if 'current' in panels:
            ax = axes[panel_idx]; panel_idx += 1
            ax.plot(t_min, v['current-density'] * 1e-4, color='C0', lw=1.2)
            ax.set_ylabel('Current density\n(A cm⁻²)')
            ax.grid(True, alpha=0.3)

        if 'pressure' in panels:
            ax = axes[panel_idx]; panel_idx += 1
            for key, label, color in [
                ('ca-outlet-pressure', 'Cathode', 'C0'),
                ('an-outlet-pressure', 'Anode',   'C1'),
            ]:
                if v.get(key) is not None:
                    ax.plot(t_min, v[key] * 1e-5, label=label, color=color, lw=1.2)
            ax.set_ylabel('Outlet pressure (bar)')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

        if 'temperature' in panels:
            ax = axes[panel_idx]; panel_idx += 1
            _temp_series = [
                ('ca-inlet-temperature',       'Gas cathode', 'C0', '-'),
                ('an-inlet-temperature',       'Gas anode',   'C1', '--'),
                ('cell-temperature',           'Cell',        'C2', '-.'),
                ('inlet-cooling-temperature',  'Coolant in',  'C3', '-'),
                ('outlet-cooling-temperature', 'Coolant out', 'C4', '--'),
            ]
            for key, label, color, ls in _temp_series:
                if v.get(key) is not None:
                    ax.plot(t_min, v[key] - 273.15,
                            label=label, color=color, ls=ls, lw=1.2)
            ax.set_ylabel('Temperature (°C)')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

        if 'rh' in panels:
            ax = axes[panel_idx]; panel_idx += 1
            ax2 = ax.twinx()
            for rh_key, T_key, label, color in [
                ('ca-inlet-rh', 'ca-inlet-temperature', 'Cathode', 'C0'),
                ('an-inlet-rh', 'an-inlet-temperature', 'Anode',   'C1'),
            ]:
                rh = v.get(rh_key)
                if rh is None:
                    continue
                ax.plot(t_min, rh * 100., label=label, color=color, lw=1.2)
                T_gas = v.get(T_key) if v.get(T_key) is not None else v.get('cell-temperature')
                if T_gas is not None:
                    ax2.plot(t_min, _dewpoint_C(rh, T_gas),
                             color=color, ls='--', lw=0.9, alpha=0.7)
            ax.set_ylabel('RH (%)')
            ax2.set_ylabel('Dew point (°C)', color='grey')
            ax2.tick_params(axis='y', labelcolor='grey')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)

        if 'stoichiometry' in panels:
            ax = axes[panel_idx]; panel_idx += 1
            for key, label, color in [
                ('ca-stoichiometry', 'Cathode', 'C0'),
                ('an-stoichiometry', 'Anode',   'C1'),
            ]:
                if v.get(key) is not None:
                    ax.plot(t_min, v[key], label=label, color=color, lw=1.2)
            ax.set_ylabel('Stoichiometry')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlim([0,t_min[-1]])
        axes[-1].set_xlabel('Time (min)')
        fig.tight_layout()
        return fig, axes

    # ── segment-based construction ─────────────────────────────────────────────

    @staticmethod
    def _profiles_from_segments(segments) -> dict:
        """Build ``{channel: PiecewiseProfile}`` from a list of :class:`CycleSegment`."""
        levels: dict = {}
        for seg in segments:
            levels.update(seg.levels)

        channels: set = set()
        for seg in segments:
            channels.update(seg.variables.keys())

        profiles: dict = {}
        for ch in channels:
            prof_segs = []
            t = 0.
            last_raw = None
            for seg in segments:
                t_end = t + seg.dwell
                v = seg.variables.get(ch)
                if v is not None:
                    kind, raw_val = v
                    last_raw = raw_val
                else:
                    if last_raw is None:
                        t = t_end
                        continue
                    kind = 'const'
                    raw_val = last_raw
                prof_segs.append((kind, raw_val, t_end))
                t = t_end
            if prof_segs:
                profiles[ch] = PiecewiseProfile(prof_segs, levels)
        return profiles

    @classmethod
    def from_segments(cls, segments, time_step: float = 1.0) -> 'LoadCycle':
        """Build a :class:`LoadCycle` from a list of :class:`CycleSegment` objects.

        Variables not defined in some segments are carried forward from the
        last segment that defined them.

        Parameters
        ----------
        segments : list of CycleSegment
        time_step : float
        """
        duration = sum(seg.dwell for seg in segments)
        profiles = cls._profiles_from_segments(segments)

        ca_kwargs = {v: profiles[k] for k, v in _CA_KEY_MAP.items() if k in profiles}
        an_kwargs = {v: profiles[k] for k, v in _AN_KEY_MAP.items() if k in profiles}

        cycle = cls(
            duration=duration,
            time_step=time_step,
            current_density=profiles.get('current-density'),
            cell_temperature=profiles.get('cell-temperature'),
            inlet_cooling_temperature=profiles.get('inlet-cooling-temperature'),
            outlet_cooling_temperature=profiles.get('outlet-cooling-temperature'),
            ca=DynamicSideConditions(**ca_kwargs) if ca_kwargs else None,
            an=DynamicSideConditions(**an_kwargs) if an_kwargs else None,
        )
        cycle._segments = list(segments)
        return cycle

    def __add__(self, other):
        if isinstance(other, CycleSegment) and hasattr(self, '_segments'):
            return LoadCycle.from_segments(self._segments + [other])
        return NotImplemented
