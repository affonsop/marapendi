Time-series simulations
========================

Two strategies are available for time-series data:

* **Quasi-steady (vectorised)** — each time step is treated as an independent
  steady-state point.  Appropriate when the MEA dynamics are fast compared to the
  load changes.
* **Transient (ODE integration)** — coupled ODEs for MEA temperature and
  through-plane membrane water-content profile.  Required when step responses,
  warm-up transients, or dynamic HFR are of interest.

Both strategies accept conditions defined directly from an experimental CSV log.

.. note::

   This guide builds the time-varying ``conditions`` callable by hand, which is
   the right tool when you already have one specific measured log to replay.
   If instead you want to *construct* a load profile from named steps, or
   drive the cell with one of the standardised ID-FAST / FC-DLC (NEDC) cycles,
   see :doc:`load_cycles` — it covers
   :class:`~marapendi.simulation.load_cycles.LoadCycle`, which does the same
   job as the hand-written callables below with vectorised evaluation and
   automatic ODE-solver breakpoints at every step change.

Loading conditions from a CSV file
-----------------------------------

Test-bench software typically exports a semicolon-delimited file with one row per
time step.  The columns you need — current, voltage, temperature, pressure, RH,
and stoichiometry — must be extracted and unit-converted before building
:class:`~marapendi.simulation.conditions.CellConditions`.

.. code-block:: python

    import numpy as np
    import pandas as pd
    import marapendi as mrpd

    CELL_AREA = 25e-4   # m²

    raw = pd.read_csv("data/test_bench_log.csv", sep=";", decimal=".",
                      skiprows=6)   # skip the instrument header

    # Rename to convenient names and convert units
    df = pd.DataFrame({
        "t":    raw["Time(s)"].values,
        "i":    raw["I_Pile(A)"].values / CELL_AREA,          # A → A/m²
        "V":    raw["U_Pile(V)"].values,
        "T_K":  raw["T_pile(°C)"].values + 273.15,            # °C → K
        "p_ca": raw["P_Air_Out(bara)"].values * 1e5,          # bara → Pa
        "p_an": raw["P_h2_out(bara)"].values * 1e5,
        "rh_ca": raw["RH_Air_calc(%)"].values / 100.,         # % → fraction
        "rh_an": raw["RH_h2_calc(%)"].values / 100.,
        "st_ca": raw["Stoeckio_air_calc"].values,
        "st_an": raw["Stoeckio_h2_calc"].values,
    })

    # Drop transient start-up rows and rows with zero current
    df = df[(df["i"] > 50)].reset_index(drop=True)

    conditions = mrpd.CellConditions(
        current_density=df["i"].values,
        cell_temperature=df["T_K"].values,
        ca=mrpd.SideConditions(
            inlet_temperature=df["T_K"].values,
            outlet_pressure=df["p_ca"].values,
            dry_o2_mole_fraction=0.21,
            inlet_relative_humidity=df["rh_ca"].values,
            stoichiometry=df["st_ca"].values,
        ),
        an=mrpd.SideConditions(
            inlet_temperature=df["T_K"].values,
            outlet_pressure=df["p_an"].values,
            dry_h2_mole_fraction=1.0,
            inlet_relative_humidity=df["rh_an"].values,
            stoichiometry=df["st_an"].values,
        ),
    )

All fields in :class:`~marapendi.simulation.conditions.CellConditions` and
:class:`~marapendi.simulation.conditions.SideConditions` accept numpy arrays; the
solver evaluates all N time steps in a single vectorised call.

Quasi-steady simulation
------------------------

Pass the vectorised ``conditions`` to the explicit model.  Each row is evaluated
independently — no ODE, no iteration between steps:

.. code-block:: python

    liq     = mrpd.DarcyTransportModel(J_function_exponent=2)
    ionomer = mrpd.PFSAIonomer(equivalent_weight=1100, dry_density=1980)
    # … (build cell as in the polarization-curve guide) …

    model = mrpd.ExplicitSteadyStateModel()
    state = model.solve(cell, conditions,
                        model.set_initial_conditions(cell, conditions))

    # state.cell_voltage — simulated voltage at each time step
    # state.membrane.water_content — mean membrane λ at each time step
    hfr = model.voltage_model.high_frequency_resistance(cell, state)

    import matplotlib.pyplot as plt
    t_h = df["t"].values / 3600
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
    axes[0].plot(t_h, df["V"], "k.", ms=2, label="Measured")
    axes[0].plot(t_h, state.cell_voltage, "C0-", lw=1, label="Simulated")
    axes[0].set_ylabel("Cell voltage (V)")
    axes[0].legend()
    axes[1].plot(t_h, hfr * 1e4, "C2-", lw=1)
    axes[1].set_ylabel("HFR (mΩ cm²)")
    axes[1].set_xlabel("Time (h)")

.. note::

   For a quasi-steady simulation you need only one ``cell`` object; the cell
   parameters are constant.  All time-varying information lives in
   ``conditions``.

Transient simulation — constant conditions
-------------------------------------------

:class:`~marapendi.models.base.transient.TransientModel` integrates the
MEA-temperature ODE and the through-plane membrane water-content
diffusion–convection PDE via :func:`scipy.integrate.solve_ivp`.  Pass a single
:class:`~marapendi.simulation.conditions.CellConditions` for constant-load
operation:

.. code-block:: python

    from marapendi.models.base.transient import TransientModel

    T = 353.15
    cond_0 = mrpd.CellConditions(
        current_density=np.atleast_1d(1e4),   # A/m²
        cell_temperature=T,
        ca=mrpd.SideConditions(
            inlet_temperature=T, outlet_pressure=1.5e5,
            dry_o2_mole_fraction=0.21, inlet_relative_humidity=0.5,
            stoichiometry=2.0,
        ),
        an=mrpd.SideConditions(
            inlet_temperature=T, outlet_pressure=1.5e5,
            dry_h2_mole_fraction=1.0, inlet_relative_humidity=0.5,
            stoichiometry=1.5,
        ),
    )

    tr_model = TransientModel(n_memb_mesh=5)
    _, x0 = tr_model.set_initial_conditions(cell, cond_0)
    sol = tr_model.solve(cell, cond_0, t_span=(0, 600), x0=x0)

The state vector ``x`` is laid out as ``[T_MEA, λ_1, …, λ_n]`` where ``n =
n_memb_mesh``.  ``sol.y`` follows the SciPy ``solve_ivp`` convention.

Transient simulation — time-varying conditions
-----------------------------------------------

Pass a callable ``conditions(t)`` instead of a fixed
:class:`~marapendi.simulation.conditions.CellConditions`:

.. code-block:: python

    I_LOW  = 5_000.   # A/m²  — before step
    I_HIGH = 20_000.  # A/m²  — after step

    def conditions(t):
        i = I_LOW if t <= 0 else I_HIGH
        return mrpd.CellConditions(
            current_density=np.atleast_1d(i),
            cell_temperature=T,
            ca=mrpd.SideConditions(
                inlet_temperature=T, outlet_pressure=1.5e5,
                dry_o2_mole_fraction=0.21, inlet_relative_humidity=0.5,
                stoichiometry=2.0,
            ),
            an=mrpd.SideConditions(
                inlet_temperature=T, outlet_pressure=1.5e5,
                dry_h2_mole_fraction=1.0, inlet_relative_humidity=0.5,
                stoichiometry=1.5,
            ),
        )

    _, x0_lo = tr_model.set_initial_conditions(cell, conditions(0))
    sol = tr_model.solve(cell, conditions, t_span=(0, 600), x0=x0_lo,
                         dense_output=True)

The callable is called at each ODE evaluation step, so any channel — current,
temperature, pressure, RH — can vary continuously.

.. note::

   A hand-written step function like the one above has a hard kink at
   ``t = 0`` that the ODE solver cannot see coming. For a handful of steps
   this is usually harmless, but for a cycle with many step changes (a
   driving cycle, a multi-point test sequence) prefer
   :class:`~marapendi.simulation.load_cycles.LoadCycle` (:doc:`load_cycles`):
   it reports its own kink locations via ``discontinuity_times()``, which
   ``TransientModel.solve`` uses to restart the integration cleanly at each
   one instead of stepping over it.

To interpolate test-bench log data into the transient conditions callable, build a
set of ``np.interp`` wrappers:

.. code-block:: python

    t_log = df["t"].values    # from the CSV loaded above

    def conditions_from_log(t):
        return mrpd.CellConditions(
            current_density=np.atleast_1d(np.interp(t, t_log, df["i"])),
            cell_temperature=np.interp(t, t_log, df["T_K"]),
            ca=mrpd.SideConditions(
                inlet_temperature=np.interp(t, t_log, df["T_K"]),
                outlet_pressure=np.interp(t, t_log, df["p_ca"]),
                dry_o2_mole_fraction=0.21,
                inlet_relative_humidity=np.interp(t, t_log, df["rh_ca"]),
                stoichiometry=np.interp(t, t_log, df["st_ca"]),
            ),
            an=mrpd.SideConditions(
                inlet_temperature=np.interp(t, t_log, df["T_K"]),
                outlet_pressure=np.interp(t, t_log, df["p_an"]),
                dry_h2_mole_fraction=1.0,
                inlet_relative_humidity=np.interp(t, t_log, df["rh_an"]),
                stoichiometry=np.interp(t, t_log, df["st_an"]),
            ),
        )

Dense output and diagnostics
-----------------------------

With ``dense_output=True``, ``sol.sol(t)`` returns the state vector interpolated
at any time.  Use
:meth:`~marapendi.models.base.transient.TransientModel.evaluate` to
compute the full :class:`~marapendi.simulation.state.CellState` diagnostics on a
fine grid:

.. code-block:: python

    t_eval = np.linspace(0, 600, 300)
    diag = tr_model.evaluate(cell, conditions, t_eval,
                              x_eval=sol.sol(t_eval))

    # diag is a CellState; all fields are arrays of shape (300,)
    diag.cell_voltage           # V
    diag.hfr                    # Ω m²  (set by evaluate, unlike the SS solver)
    diag.mea_temperature        # K
    diag.membrane.water_content # mol/mol

    # Through-plane water-content profile at each time step
    # diag.membrane.water_content_profile  — shape (n_memb_mesh, 300)

    fig, ax = plt.subplots()
    t_min = t_eval / 60
    ax.plot(t_min, diag.cell_voltage, label="Voltage")
    ax2 = ax.twinx()
    ax2.plot(t_min, diag.hfr * 1e4, "C2--", label="HFR")
