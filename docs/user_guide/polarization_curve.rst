Simulating a polarization curve
================================

This guide walks through a complete transient simulation of a PEMFC
polarization curve using :class:`~marapendi.models.transient.TransientCellModel`.
The approach mimics a galvanostatic step protocol: current density is held
constant at each operating point while the internal state (water content,
temperature, gas concentrations, liquid saturation) reaches near-steady state,
then the cell voltage is recorded.

The full runnable version lives in ``notebooks/simulate_polarization_curve.ipynb``.

----

Cell assembly
-------------

Components are pure dataclasses that hold geometry and material parameters.
Computation is delegated to stateless model objects passed to :class:`~marapendi.components.cell.Cell`.

.. code-block:: python

    import numpy as np
    import cantera as ct
    from scipy.integrate import solve_ivp
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

    cl = mrpd.PtCCatalystLayer(
        thickness=10e-6,
        bulk_density=2010.,
        bulk_specific_heat_capacity=710.,
        bulk_thermal_conductivity=0.25,
        L_Pt=0.3e-2,
        wt_Pt=0.4,
        ic_ratio=0.7,
        ecsa=45e3,
        ionomer=mrpd.Nafion_N21X,
        r_C=25e-9,
        K_abs=1e-13,
        theta_contact=95,
        reaction=orr_kinetics,
    )

    gdl = mrpd.PorousLayer(
        thickness=160e-6,
        eps_p=0.72,
        bulk_density=440.,
        bulk_specific_heat_capacity=710.,
        bulk_thermal_conductivity=1.24,
        K_abs=1e-12,
        theta_contact=115.,
        tort=3,
    )

    cell = mrpd.Cell(
        area=25e-4,
        electrical_resistance=30e-7,
        thermal_resistance=2e-4,
        memb_model=mrpd.PFSAModel(),
        cl_model=mrpd.PtCCatalystLayerModel(),
        ca=mrpd.CellSide(
            cl=cl, gdl=gdl,
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        an=mrpd.CellSide(
            cl=cl, gdl=gdl,
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        memb=mrpd.Nafion_N212,
    )

    model = mrpd.TransientCellModel(cell=cell)

The layer stack and their indices can be inspected::

    for layer in cell.layers:
        print(f"[{layer.ix}] {layer.name}  {layer.thickness*1e6:.0f} µm")

    # [0] anode channel    1000 µm
    # [1] anode GDL         160 µm
    # [2] anode CL           10 µm
    # [3] membrane           50 µm
    # [4] cathode CL         10 µm
    # [5] cathode GDL       160 µm
    # [6] cathode channel  1000 µm

----

Initial conditions
------------------

:meth:`~marapendi.models.transient.TransientCellModel.initial_state` constructs
the normalised flat state vector expected by ``solve_ivp`` from operating
conditions.  It assigns the correct gas composition to each layer depending on
which side of the membrane it sits on, and guards missing species with a small
epsilon to avoid ``log(0)`` in the Nernst equation at *t* = 0.

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

.. code-block:: python

    y0 = model.initial_state(
        T=353.15,       # K  (80 °C)
        p=1.5e5,        # Pa (1.5 bara)
        rh=0.7,         # relative humidity
        lmbd=10.0,      # initial water content [mol/mol]
        s=0.05,         # initial liquid saturation
        ca_dry_o2=0.21, # cathode: air
        an_dry_h2=1.0,  # anode: pure H₂
    )

----

Step-current polarization curve
---------------------------------

Each current step is integrated with BDF.  The final state of one step
becomes the initial state of the next, so the cell progressively hydrates
and warms — matching a real galvanostatic sweep.

.. code-block:: python

    def postprocess(model, y_flat, i_density):
        """Populate a CellState with voltage and thermodynamic fields."""
        x = (y_flat.reshape(model.n_layers, model.n_variables, -1)
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
        sol = solve_ivp(
            fun=lambda t, y: model.rates_of_change(y[:, np.newaxis], i=i_density)[:, 0],
            t_span=(0., T_STEP),
            y0=y,
            method='BDF',
            max_step=10.,
            rtol=1e-3,
            atol=1e-6,
        )
        y = sol.y[:, -1]
        results.append(postprocess(model, y[:, np.newaxis], i_density))

    V_cell  = np.array([float(s.V_cell)  for s in results])
    eta_ohm = np.array([float(s.eta_ohm) for s in results])
    eta_act = np.array([float(s.eta_act) for s in results])

----

Composing multiple cells
-------------------------

:class:`~marapendi.models.model.BaseModel` composes any number of submodels
into a single ODE system whose state vector is the concatenation of each
submodel's state vector.  Each submodel only needs to expose
``rates_of_change(x, **inputs)`` and an ``n_states`` integer.

This is useful for simulating a stack of cells at different operating
points, or for coupling a cell model to an auxiliary system.

.. code-block:: python

    model_a = mrpd.TransientCellModel(cell=cell_a)
    model_b = mrpd.TransientCellModel(cell=cell_b)

    i_a = lambda t: 5000.    # A/m² — constant load on cell A
    i_b = lambda t: 10000.   # A/m² — different load on cell B

    base = mrpd.BaseModel(
        submodels={'cell_a': model_a, 'cell_b': model_b},
        input_fns={
            'cell_a': lambda t: {'i': i_a(t)},
            'cell_b': lambda t: {'i': i_b(t)},
        },
    )

    y0 = base.initial_state(
        cell_a={'T': 353.15, 'p': 1.5e5, 'rh': 0.7},
        cell_b={'T': 343.15, 'p': 1.0e5, 'rh': 0.6},
    )

    sol = solve_ivp(
        base.rates_of_change,   # signature (t, x) — matches solve_ivp directly
        t_span=(0., 200.),
        y0=y0,
        method='BDF',
        max_step=10.,
    )

    # Split the solution back into per-cell arrays
    states = base.split_state(sol.y)
    # states['cell_a'] shape: (model_a.n_states, n_timepoints)
    # states['cell_b'] shape: (model_b.n_states, n_timepoints)

``BaseModel`` itself satisfies the same protocol (it has ``rates_of_change``
and ``n_states``), so models can be nested arbitrarily.
