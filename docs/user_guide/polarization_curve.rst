:orphan:

.. note::

   This guide describes the experimental, transient-capable
   :mod:`marapendi.dynamic` package (inspired by Yang et al. 2019), which is
   under evaluation and not part of the main documented API. Module paths
   below (``marapendi.components.*``, ``marapendi.models.*``) refer to
   ``marapendi.dynamic.components.*`` / ``marapendi.dynamic.models.*``.

Simulating a polarization curve
================================

This guide covers two complementary strategies for computing a PEMFC
polarization curve:

1. **Transient sweep** (IVP) — current density is held constant at each
   operating point while the internal state (water content, temperature, gas
   concentrations, liquid saturation) is integrated until near-steady state,
   then the cell voltage is recorded.

2. **Steady-state sweep** — :meth:`~marapendi.models.model.BaseModel.solve_steady_state`
   solves ``rates_of_change(t, x) = 0`` directly for each operating point,
   skipping time integration entirely.  Each solve is warm-started from the
   previous solution.

Both strategies are provided as runnable notebooks:

* ``notebooks/simulate_polarization_curve.ipynb`` — IVP sweep, no channel dynamics.
* ``notebooks/simulate_polarization_curve_channel.ipynb`` — IVP sweep with dynamic
  channel gas concentrations.
* ``notebooks/simulate_polarization_curve_steady_state.ipynb`` — steady-state sweep
  with channel dynamics; includes a solver benchmark (Section 8).

Design overview
---------------

The simulation stack has three layers:

* **Components** (:class:`~marapendi.components.cell.Cell`,
  :class:`~marapendi.components.catalyst_layers.PtCCatalystLayer`, …)
  — pure dataclasses holding geometry and material parameters.  No equations.

* :class:`~marapendi.models.transient.TransientCellModel`
  — the ODE engine.  Holds a :class:`~marapendi.components.cell.Cell`
  and a ``current_density`` field (scalar or callable).  Physics models are
  resolved from its parent ``CellBaseModel`` via the injected ``base_model``
  reference.

* :class:`~marapendi.models.cell_base_model.CellBaseModel`
  — owns all five physics strategy objects as named fields and wires the
  ``TransientCellModel`` into its ``submodels`` dict automatically.
  Provides :meth:`~marapendi.models.model.BaseModel.solve` for one-line integration.

----

Cell assembly
-------------

.. important::

   Each side **must use its own** :class:`~marapendi.components.porous_layers.PorousLayer`
   instance.  Sharing one object causes its ``.ix`` index to be overwritten
   by the second assignment, leaving the anode GDL slot unwritten (zero gas
   pressure) in the initial state.

.. code-block:: python

    import numpy as np
    import marapendi as mrpd

    orr_kinetics = mrpd.ElectrochemicalReaction(
        reference_exchange_current_density=2.47e-8 * 10e-6,
        activation_energy=67e6,
        reaction_order=0.54,
        reference_activity=1.,
        reference_temperature=353.15,
        number_of_electrons=2,
        charge_transfer_coeff=1,
    )

    ca_cl = mrpd.PtCCatalystLayer(
        thickness=10e-6, bulk_density=2010., bulk_specific_heat_capacity=710.,
        bulk_thermal_conductivity=0.25, L_Pt=0.3e-2, wt_Pt=0.4, ic_ratio=0.7,
        ecsa=45e3, ionomer=mrpd.Nafion_N21X, r_C=25e-9, K_abs=1e-13,
        theta_contact=95, reaction=orr_kinetics,
    )
    an_cl = mrpd.PtCCatalystLayer(   # separate instance — same parameters
        thickness=10e-6, bulk_density=2010., bulk_specific_heat_capacity=710.,
        bulk_thermal_conductivity=0.25, L_Pt=0.3e-2, wt_Pt=0.4, ic_ratio=0.7,
        ecsa=45e3, ionomer=mrpd.Nafion_N21X, r_C=25e-9, K_abs=1e-13,
        theta_contact=95, reaction=orr_kinetics,
    )

    gdl_ca = mrpd.PorousLayer(
        thickness=160e-6, eps_p=0.72, bulk_density=440.,
        bulk_specific_heat_capacity=710., bulk_thermal_conductivity=1.24,
        K_abs=1e-12, theta_contact=115., tort=3,
    )
    gdl_an = mrpd.PorousLayer(   # separate instance for the same reason
        thickness=160e-6, eps_p=0.72, bulk_density=440.,
        bulk_specific_heat_capacity=710., bulk_thermal_conductivity=1.24,
        K_abs=1e-12, theta_contact=115., tort=3,
    )

    # Cell holds geometry and materials only — no physics model objects.
    cell = mrpd.Cell(
        area=25e-4, electrical_resistance=30e-7, thermal_resistance=2e-4,
        ca=mrpd.CellSide(
            cl=ca_cl, gdl=gdl_ca,
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        an=mrpd.CellSide(
            cl=an_cl, gdl=gdl_an,
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        memb=mrpd.Nafion_N212,
    )

The layer stack and their indices::

    model = mrpd.TransientCellModel(cell=cell)
    for layer in cell.layers:
        print(f"[{layer.ix}] {layer.name}  {layer.thickness*1e6:.0f} µm")

    # [0] porous layer  1000 µm   ← anode channel
    # [1] porous layer   160 µm   ← anode GDL
    # [2] porous layer    10 µm   ← anode CL
    # [3] porous layer    50 µm   ← membrane (Nafion N212)
    # [4] porous layer    10 µm   ← cathode CL
    # [5] porous layer   160 µm   ← cathode GDL
    # [6] porous layer  1000 µm   ← cathode channel

----

Building the CellBaseModel
--------------------------

:class:`~marapendi.models.cell_base_model.CellBaseModel` owns the five
physics strategy objects as named fields.  On construction it wires the
``TransientCellModel`` into ``submodels`` automatically, registers
:meth:`~marapendi.models.transient.TransientCellModel.get_inputs` so the
``current_density`` field flows into the ODE dispatcher without any manual
``input_fns`` dict, and injects itself into
``transient_transport_model.base_model`` so the ODE engine can resolve
physics objects.

.. code-block:: python

    base = mrpd.CellBaseModel(
        transient_transport_model=mrpd.TransientCellModel(
            cell=cell,
            current_density=0.,   # will be updated each step in the loop
        ),
        memb_model=mrpd.PFSAModel(),
        cl_model=mrpd.PtCCatalystLayerModel(),
        gas_diffusion_model=mrpd.PorousGasResistanceModel(),
        darcy_transport_model=mrpd.DarcyTransportModel(),
        voltage_model=mrpd.VoltageModel(),
    )

    model = base.transient_transport_model   # shorthand for post-processing

----

Initial conditions
------------------

:meth:`~marapendi.models.cell_base_model.CellBaseModel.initial_state`
accepts flat keyword arguments and forwards them to
:meth:`~marapendi.models.transient.TransientCellModel.initial_state`.
Ionomer water content is initialised at the equilibrium value for the
average inlet relative humidity; liquid saturation starts at zero.

.. code-block:: python

    y0 = base.initial_state(
        cell_temperature=353.15,   # K  (80 °C)
        cell_pressure=1.5e5,       # Pa (1.5 bara)
        ca_rh=0.7,                 # cathode inlet RH
        an_rh=0.7,                 # anode inlet RH
        ca_dry_o2=0.21,            # cathode dry-gas O₂ fraction (air)
        an_dry_h2=1.0,             # anode dry-gas H₂ fraction (pure H₂)
    )

The state vector has ``n_layers × n_variables = 7 × 7 = 49`` elements:

.. list-table::
   :header-rows: 1
   :widths: 10 15 15 60

   * - Index
     - Symbol
     - Unit
     - Description
   * - 0
     - λ
     - mol/mol
     - Ionomer water content
   * - 1
     - T
     - K
     - Temperature
   * - 2
     - c\ :sub:`O₂`
     - kmol/m³
     - O₂ concentration
   * - 3
     - c\ :sub:`N₂`
     - kmol/m³
     - N₂ concentration
   * - 4
     - c\ :sub:`H₂`
     - kmol/m³
     - H₂ concentration
   * - 5
     - c\ :sub:`H₂O`
     - kmol/m³
     - Water vapour concentration
   * - 6
     - s
     - —
     - Liquid water saturation

----

Step-current polarization curve
---------------------------------

Set ``current_density`` directly on the ``TransientCellModel`` before each
step — no mutable containers or ``input_fns`` dicts needed.
:meth:`~marapendi.models.model.BaseModel.solve` wraps ``scipy.integrate.solve_ivp``
with BDF as the default method.

.. code-block:: python

    def postprocess(y_flat, i_density):
        """Populate a CellState with voltage and thermodynamic fields."""
        cell_y = base.split_state(y_flat)['cell']
        x = (cell_y.reshape(model.n_layers, model.n_variables, -1)
             * model.norm_factor[..., np.newaxis])
        state = model._compute_derived_quantities(x, i_density)
        model._compute_voltage(state)
        return state

    CURRENT_DENSITIES = np.array(
        [200, 500, 1000, 2000, 4000, 7000, 10000, 15000, 20000]
    )  # A/m²
    T_STEP = 100.0   # s per current step

    results = []
    y = y0.copy()

    for i_density in CURRENT_DENSITIES:
        model.current_density = float(i_density)
        sol = base.solve(y, t_span=(0., T_STEP), max_step=10.)
        y = sol.y[:, -1]
        results.append(postprocess(y[:, np.newaxis], i_density))

    V_cell  = np.array([s.V_cell.item()  for s in results])
    eta_ohm = np.array([s.eta_ohm.item() for s in results])
    eta_act = np.array([s.eta_act.item() for s in results])

For a time-varying current (e.g. a load transient), assign a callable::

    model.current_density = lambda t: 5000. if t < 50 else 10000.

----

Transient diagnostics
---------------------

For a single fixed-current run, create a dedicated ``CellBaseModel`` with a
constant ``current_density`` and pass a ``t_eval`` grid:

.. code-block:: python

    base_diag = mrpd.CellBaseModel(
        transient_transport_model=mrpd.TransientCellModel(
            cell=cell, current_density=10000.,   # 1 A/cm²
        ),
        memb_model=mrpd.PFSAModel(),
        cl_model=mrpd.PtCCatalystLayerModel(),
        gas_diffusion_model=mrpd.PorousGasResistanceModel(),
        darcy_transport_model=mrpd.DarcyTransportModel(),
        voltage_model=mrpd.VoltageModel(),
    )

    t_eval = np.linspace(0, 200, 201)
    sol = base_diag.solve(y0, t_span=(0., 200.), max_step=2., t_eval=t_eval)
    state = postprocess(sol.y, 10000.)

    # cell voltage over time
    plt.plot(sol.t, state.V_cell)

----

Composing multiple cells
-------------------------

:class:`~marapendi.models.model.BaseModel` composes any number of submodels
into a single ODE.  Each submodel only needs to expose
``rates_of_change(x, **inputs)`` and ``n_states``.
Since :class:`~marapendi.models.cell_base_model.CellBaseModel` satisfies both,
two cells can be composed directly:

.. code-block:: python

    base_a = mrpd.CellBaseModel(
        transient_transport_model=mrpd.TransientCellModel(
            cell=cell_a, current_density=5000.,
        ),
        memb_model=mrpd.PFSAModel(), ...
    )
    base_b = mrpd.CellBaseModel(
        transient_transport_model=mrpd.TransientCellModel(
            cell=cell_b, current_density=10000.,
        ),
        memb_model=mrpd.PFSAModel(), ...
    )

    composed = mrpd.BaseModel(submodels={'a': base_a, 'b': base_b})

    y0 = composed.initial_state(a=ic_a, b=ic_b)
    sol = composed.solve(y0, t_span=(0., 200.))

    states = composed.split_state(sol.y)
    # states['a'] shape: (base_a.n_states, n_timepoints)
    # states['b'] shape: (base_b.n_states, n_timepoints)

``BaseModel`` itself exposes ``rates_of_change`` and ``n_states``, so
composed models can be nested arbitrarily.

----

Steady-state polarization curve
---------------------------------

:meth:`~marapendi.models.model.BaseModel.solve_steady_state` wraps
``scipy.optimize.root`` (MINPACK HYBRD by default) to find ``x`` such that
``rates_of_change(t, x) = 0``.  The evaluation time *t* is used only to
resolve time-varying inputs (current density, inlet conditions); it has no
effect on the solution when those are fixed scalars.

.. code-block:: python

    CURRENT_DENSITIES = np.linspace(0, 20000, 101)   # A/m²

    V_arr   = np.full(len(CURRENT_DENSITIES), np.nan)
    y_store = np.full((len(CURRENT_DENSITIES), len(y0)), np.nan)

    y = y0.copy()
    for k, i_k in enumerate(CURRENT_DENSITIES):
        model.current_density      = float(i_k)   # plain float is safe here
        conditions.current_density = float(i_k)   # CellConditions.at() handles floats

        sol = base.solve_steady_state(y, t=0.)
        if not sol.success:
            print(f"Step {k}: solver did not converge — stopping.")
            break

        st = base.postprocess(sol.y, i_density=float(i_k))
        V_k = float(st.V_cell[0])
        if V_k <= 0.:
            break

        V_arr[k]   = V_k
        y_store[k] = sol.y[:, 0]
        y          = sol.y[:, 0]   # warm-start next step

The return value of ``solve_steady_state`` is a
:class:`types.SimpleNamespace` whose ``.y`` has shape ``(n_states, 1)``,
making it a drop-in replacement for a single-column ``sol.y`` slice from
``solve_ivp``:

.. code-block:: python

    # IVP style (single-column slice)
    st = base.postprocess(sol_ivp.y[:, -1:], i_density=float(i_k))

    # Steady-state style — identical call
    st = base.postprocess(sol_ss.y, i_density=float(i_k))

**Advantages over IVP:**  each operating point is solved in one shot with
no integration time, giving a ~10–100× wall-clock speedup (exact ratio
depends on the integration window and stiffness).

**Limitations:**  the root finder may not converge near the limiting current
density where the Jacobian becomes ill-conditioned, and it does not capture
transient overshoot or hysteresis.

Assigning float current density after construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both :class:`~marapendi.components.operating_conditions.CellConditions` and
:class:`~marapendi.models.transient.TransientCellModel` accept either a plain
``float`` or a ``callable f(t) → float`` for ``current_density``.  In a
sweep loop it is common to assign a raw float **after** construction:

.. code-block:: python

    model.current_density      = float(i_k)   # updates TransientCellModel
    conditions.current_density = float(i_k)   # updates CellConditions

``CellConditions.at(t)`` detects whether ``current_density`` is callable via
:func:`callable` and uses the float directly when it is not, so no explicit
wrapping (``lambda _t: i_k``) is needed.  Constructing with a callable and
reassigning with a float mid-loop is equally valid.

NaN-safe step selection
~~~~~~~~~~~~~~~~~~~~~~~

When only a subset of steps converge (or the IVP terminates early), the
voltage array ``V_arr`` contains ``np.nan`` for unreached steps.
Use :func:`numpy.nanargmin` — not :func:`numpy.argmin` — to find the
operating point closest to a target voltage:

.. code-block:: python

    # WRONG: np.argmin propagates NaN and may return the first NaN index
    idx = np.argmin(np.abs(V_arr - 0.6))

    # CORRECT: nanargmin ignores NaN entries
    idx = np.nanargmin(np.abs(V_arr - 0.6))

If ``np.argmin`` is used and ``V_arr`` contains NaN, it typically returns
the index of the **first NaN entry** (whose ``|NaN - 0.6| = NaN`` is treated
as negative infinity by some BLAS implementations).  This causes
``(idx + 1) * T_STEP > sol_full.t[-1]``, so the state snapshot falls back to
``sol_full.t[-1]`` — the simulation termination point at high current —
rather than the intended 0.6 V operating point.
