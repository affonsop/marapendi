Models
======

Orchestration classes that combine a :class:`~marapendi.cell.fuelcell.FuelCell` and a
:class:`~marapendi.simulation.state.CellState` to compute the cell's behaviour:
membrane water balance, gas transport, voltage and thermal sub-models.

Steady-state models
-------------------

Two top-level steady-state models are available, differing in how MEA temperature
is handled:

- :class:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel` —
  MEA temperature is estimated analytically (one forward pass).
- :class:`~marapendi.models.base.implicit_steady_state.ImplicitSteadyStateModel` —
  cell voltage and MEA temperature are solved self-consistently via a
  vectorised elementwise secant iteration (:func:`scipy.optimize.newton`).

Both models use :class:`~marapendi.models.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise`
by default.  This model solves a 1-D water-content profile with boundary conditions
derived from a piecewise-linear regression of the PFSA equilibrium isotherm RH(λ).
The first-order linear approximation of the isotherm used in Affonso Nobrega et al.
(2026) is available as :class:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel`.

All models share the same two-step API::

    model = ExplicitSteadyStateModel()          # or ImplicitSteadyStateModel(...)
    conditions = CellConditions(
        current_density=np.linspace(1e3, 2e4, 20),
        cell_temperature=353.15,
        ca=SideConditions(outlet_pressure=1.5e5, dry_o2_mole_fraction=0.21, ...),
        an=SideConditions(outlet_pressure=1.5e5, dry_h2_mole_fraction=1.0, ...),
    )
    state = model.set_initial_conditions(cell, conditions)
    state = model.solve(cell, conditions, state)
    # state.cell_voltage, state.mea_temperature, … are now populated

Operating conditions are described by :class:`~marapendi.simulation.conditions.CellConditions`
and :class:`~marapendi.simulation.conditions.SideConditions`.

Transient model
---------------

:class:`~marapendi.models.base.transient.TransientModel` integrates coupled ODEs for MEA
temperature and membrane water-content profile (Ferrara et al., 2018) via
:func:`scipy.integrate.solve_ivp`.  The ``solve()`` method auto-attaches a
``diagnostics`` :class:`~marapendi.simulation.state.CellState` (voltage, HFR, water
contents, liquid saturation) evaluated at each internal time step::

    tr_model = TransientModel(n_memb_mesh=5)
    sol = tr_model.solve(cell, conditions, t_span=(0, 3600))
    # sol.diagnostics.cell_voltage  — voltage at each ODE time step
    # sol.diagnostics.hfr           — HFR at each ODE time step
    # sol.diagnostics.membrane.water_content_profile — shape (n_mesh, n_t)

    # Re-sample at arbitrary times using dense output:
    t_eval = np.linspace(0, 3600, 100)
    sol = tr_model.solve(cell, conditions, t_span=(0, 3600), dense_output=True,
                         compute_diagnostics=False)
    diag = tr_model.evaluate(cell, conditions, t_eval, x_eval=sol.sol(t_eval))


.. automodule:: marapendi.models.base.explicit_steady_state
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.models.base.implicit_steady_state
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.models.base.transient
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.models.water_balance.water_balance
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.models.water_balance.membrane_pwl
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.models.water_balance.membrane
   :members:
   :undoc-members:
   :show-inheritance:
