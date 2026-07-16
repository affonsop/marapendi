Defining load cycles
=====================

A :class:`~marapendi.simulation.conditions.CellConditions` describes the cell at
one instant (or, if its fields are arrays, a batch of independent instants — see
:doc:`polarization_curve`). A **load cycle** is different: it is a single object
that describes how every operating variable evolves *continuously* over time, so
it can be handed straight to
:meth:`~marapendi.models.base.transient.TransientModel.solve` and driven
through minutes or hours of duty in one call.

This guide builds that idea up in stages: a single time-varying profile, a
cycle built by hand from named segments, the two standardised automotive
cycles that ship with **marapendi**, and finally how to drive a cycle from
measured/logged data in a CSV file.

Why not just write a Python function?
--------------------------------------

Nothing stops you from writing ``def conditions(t): ...`` and returning a
:class:`~marapendi.simulation.conditions.CellConditions` by hand (see
:doc:`time_series` for that pattern) — it is perfectly valid and
:meth:`~marapendi.models.base.transient.TransientModel.solve` only
requires *something* callable as ``conditions(t)``. The
:class:`~marapendi.simulation.load_cycles.LoadCycle` machinery described here
exists to make that function easier to build and safer to integrate:

* Every channel (current density, pressures, temperatures, RH, stoichiometry,
  …) is defined independently and combined for you — no need to remember the
  full :class:`~marapendi.simulation.conditions.CellConditions` /
  :class:`~marapendi.simulation.conditions.SideConditions` signature every time.
* Piecewise-constant and ramp segments are vectorised
  (:class:`~marapendi.simulation.load_cycles.PiecewiseProfile`), so evaluating
  a cycle at an array of times — as :meth:`evaluate` and :meth:`plot` do — is
  fast.
* The cycle can report *where* it has kinks
  (:meth:`~marapendi.simulation.load_cycles.LoadCycle.discontinuity_times`), which
  ``TransientModel.solve`` uses automatically to keep the ODE solver from
  stepping over a discontinuity undetected (see :ref:`load-cycles-breakpoints`
  below).

Piecewise profiles
-------------------

The basic building block is
:class:`~marapendi.simulation.load_cycles.PiecewiseProfile`: a list of
``(kind, value, t_end)`` segments, where ``kind`` is ``'const'`` (hold *value*
until *t_end*) or ``'ramp'`` (linearly interpolate from the previous segment's
end value to *value* by *t_end*). The first segment always starts at
``t = 0``.

.. code-block:: python

    from marapendi.simulation import PiecewiseProfile

    T_profile = PiecewiseProfile([
        ('const', 344.15, 2745.),   # hold 71 °C for 2745 s
        ('ramp',  363.15, 3235.),   # ramp 71 → 90 °C over the next 490 s
        ('ramp',  344.15, 3725.),   # ramp back down over the next 490 s
        ('const', 344.15, 3925.),   # hold 71 °C for the final 200 s
    ])

    T_profile(2745.)   # 344.15  (still on the plateau)
    T_profile(2990.)   # ≈ 353.7 (midpoint of the up-ramp)

A profile is callable at a scalar or an array of times — the array form is
what :meth:`~marapendi.simulation.load_cycles.LoadCycle.get_input_vectors` and
:meth:`~marapendi.simulation.load_cycles.LoadCycle.plot` use internally.
Outside ``[0, t_end]`` of the last segment, evaluation clamps to the nearest
boundary value rather than extrapolating.

Building a cycle from named segments
--------------------------------------

For hand-built cycles,
:class:`~marapendi.simulation.load_cycles.CycleSegment` is usually more
convenient than constructing :class:`PiecewiseProfile` objects channel by
channel. Each segment declares a dwell time and the operating variables that
change during it — using the same hyphenated channel names produced by
:meth:`~marapendi.simulation.load_cycles.LoadCycle.get_input_vectors` (for
example ``ca-outlet-pressure``, ``an-inlet-rh``), or the underscored keyword
shorthand shown below. Any variable not mentioned in a segment simply carries
forward the previous segment's value:

.. code-block:: python

    from marapendi.simulation import CycleSegment, LoadCycle

    idle = CycleSegment(
        dwell=100,
        current_density=('const', 1000.),
        ca_outlet_pressure=('const', 1.4e5),
        an_outlet_pressure=('const', 1.9e5),
        ca_inlet_rh=('const', 0.5),
        an_inlet_rh=('const', 0.5),
        ca_stoichiometry=('const', 2.0),
        an_stoichiometry=('const', 1.5),
    )
    peak = CycleSegment(dwell=50, current_density=('const', 15_000.))
    ramp_down = CycleSegment(dwell=30, current_density=('ramp', 1000.))

    # Segments add up to a LoadCycle — as many times as needed:
    cycle = idle + peak + ramp_down + idle
    # equivalently:
    cycle = LoadCycle.from_segments([idle, peak, ramp_down, idle])

    cycle.duration       # 280.  (sum of the dwell times)
    cond = cycle(120.)   # CellConditions snapshot at t = 120 s

``levels`` lets you name values once and refer to them by string, which keeps
long cycles (dozens of repeated steps, as in the built-in cycles below)
readable:

.. code-block:: python

    levels = {'idle': 1000., 'peak': 15_000.}
    seg_a = CycleSegment(dwell=100, levels=levels, current_density=('const', 'idle'))
    seg_b = CycleSegment(dwell=50,  levels=levels, current_density=('const', 'peak'))
    cycle = sum([seg_a, seg_b, seg_a, seg_b, seg_a])   # 5-segment repeating cycle

Once assembled, a cycle can be inspected visually — one panel per condition
category, current density, pressures, temperatures, RH/dew point, and
stoichiometry, each shown only if the cycle actually defines it:

.. code-block:: python

    fig, axes = cycle.plot()

Built-in standardised cycles
------------------------------

Two automotive load cycles used in the PEMFC durability literature are
implemented directly as :class:`~marapendi.simulation.load_cycles.LoadCycle`
subclasses, so they can be used exactly like a hand-built cycle — no need to
transcribe step tables yourself.

ID-FAST
~~~~~~~

:class:`~marapendi.simulation.load_cycles.idfast.IDFastCycle` implements the
ID-FAST driving cycle: a 3925 s cycle made of a 2005 s low-power
cold section followed by a 1920 s high-power hot section, including a short
idle-stop period where the cathode air flow is cut (modelled as a very low
dry-O2 mole fraction) and current is drawn through an external resistance
until the voltage collapses.

.. code-block:: python

    from marapendi.simulation import IDFastCycle

    cycle = IDFastCycle(
        current_densities={'C0': 950., 'C1': 2470., 'C2': 5890.,
                            'C3': 12730., 'C4': 17480.}
    )
    cycle.duration       # 3925 s
    cycle.low_duration    # 2005 s
    cycle.high_duration   # 1920 s
    fig, axes = cycle.plot()

FC-DLC / NEDC
~~~~~~~~~~~~~~

:class:`~marapendi.simulation.load_cycles.nedc.NEDCCycle` implements the **Fuel Cell
Dynamic Load Cycle** derived from the New European Driving Cycle and
standardised by the JRC/FCH-JU (Tsotridis et al., EUR 27632 EN, 2015,
Appendix F): 35 piecewise-constant current steps over 1181 s, with every
other condition held constant.

.. code-block:: python

    from marapendi.simulation import NEDCCycle

    cycle = NEDCCycle(max_current_density=17_000.)   # 1.7 A cm⁻² at 100 % load
    cycle.duration   # 1181 s

Both constructors expose the physically meaningful parameters (power-level
current densities, cell temperature, pressures, humidities, stoichiometries)
as keyword arguments with literature-typical defaults — see the
:doc:`API reference </api/tools>` for the full parameter list.

Driving a transient solve with a cycle
-----------------------------------------

Because a :class:`~marapendi.simulation.load_cycles.LoadCycle` (built-in or
hand-assembled) is callable as ``cycle(t) -> CellConditions``, it can be
passed directly as the *conditions* argument of
:meth:`~marapendi.models.base.transient.TransientModel.solve`:

.. code-block:: python

    from marapendi.models.base.transient import TransientModel

    tr_model = TransientModel(n_memb_mesh=5)
    state0, x0 = tr_model.set_initial_conditions(cell, cycle(0))
    state = tr_model.solve(
        cell, cycle, t_span=(0, cycle.duration),
        x0=x0, dense_output=True, method='BDF', max_step=10,
        compute_diagnostics=False,
    )

See :doc:`time_series` for the full transient workflow.

.. _load-cycles-breakpoints:

Why discontinuity matters: ``discontinuity_times`` and ``breakpoints``
----------------------------------------------------------------------

A ``'const'`` step change or the corner of a ``'ramp'`` is a genuine
discontinuity in the derivative of the operating conditions. An adaptive ODE
solver's local error estimate assumes the right-hand side is smooth over the
step it just took; stepping across a discontinuous derivative invisibly can silently degrade
accuracy.

Every :class:`~marapendi.simulation.load_cycles.PiecewiseProfile` field of a
cycle records its own segment boundaries, and
:meth:`~marapendi.simulation.load_cycles.LoadCycle.discontinuity_times`
collects them (from the cycle itself and its nested ``ca``/``an`` side
conditions) into one sorted array of interior times.
:meth:`TransientModel.solve` uses this automatically: unless you pass
``breakpoints=`` explicitly, it calls ``cycle.discontinuity_times()`` and
restarts :func:`scipy.integrate.solve_ivp` at each one, stitching the
per-segment solutions back into a single result. 
This is why the ID-FAST and NEDC examples above need no
special handling despite dozens of step changes — pass ``breakpoints=[]`` to
``solve`` if you want to disable it and let the solver step through freely.

Importing a load cycle from CSV
-----------------------------------

Standardised cycles are convenient, but real logs
usually arrive as a CSV/Excel export with one row per sample and one column
per channel. Because **marapendi** already depends on ``pandas``, the
cleanest way to turn such a file into a :class:`LoadCycle` is to read it with
:func:`pandas.read_csv` and build one :class:`PiecewiseProfile` per channel
from consecutive rows.

Assume a log with columns ``time_s``, ``current_A_cm2``, ``cell_temp_C``:

.. code-block:: python

    import pandas as pd
    from marapendi.simulation import PiecewiseProfile, LoadCycle
    from marapendi.simulation.conditions import DynamicSideConditions

    log = pd.read_csv("data/idfast_log.csv")
    log = log.sort_values("time_s").reset_index(drop=True)

    def piecewise_from_log(t_s, values, kind='const'):
        """Build a PiecewiseProfile holding *values[k]* over (t_s[k], t_s[k+1]]."""
        segs = [(kind, float(v), float(t_end))
                for v, t_end in zip(values[:-1], t_s[1:])]
        return PiecewiseProfile(segs)

    i_profile = piecewise_from_log(
        log["time_s"], log["current_A_cm2"] * 1e4,   # A/cm² → A/m²
    )
    T_profile = piecewise_from_log(
        log["time_s"], log["cell_temp_C"] + 273.15, kind='ramp',
    )

    cycle = LoadCycle(
        duration=float(log["time_s"].iloc[-1]),
        current_density=i_profile,
        cell_temperature=T_profile,
        dT_cool=4.,
        ca=DynamicSideConditions(outlet_pressure=1.4e5, inlet_relative_humidity=0.5,
                                  stoichiometry=1.6, dry_o2_mole_fraction=0.21),
        an=DynamicSideConditions(outlet_pressure=1.9e5, inlet_relative_humidity=0.5,
                                  stoichiometry=1.4, dry_h2_mole_fraction=1.0),
    )

    fig, axes = cycle.plot()   # sanity-check the imported profile before solving

Use ``kind='const'`` for channels that are genuinely sampled/held (typical of
current density in a step test) and ``kind='ramp'`` for channels that vary
continuously between samples (typical of a measured temperature or pressure
trace) — mixing both within the same cycle, one call to
``piecewise_from_log`` per channel, is normal and matches how
:class:`IDFastCycle` itself builds independent profiles per channel
internally.

If your log already has one row per *condition change* rather than per
time-sample (e.g. a manually authored step table like the FC-DLC table used
by :class:`NEDCCycle`), it is often just as easy to build
:class:`CycleSegment` objects directly from the DataFrame rows and sum them,
which also lets different channels change on different rows:

.. code-block:: python

    steps = pd.read_csv("data/my_step_table.csv")  # columns: dwell_s, current_A_cm2

    segments = [
        CycleSegment(dwell=row.dwell_s, current_density=('const', row.current_A_cm2 * 1e4))
        for row in steps.itertuples()
    ]
    cycle = LoadCycle.from_segments(segments)
