Models
======

Orchestration classes that combine a :class:`~marapendi.cell.Cell` and a
:class:`~marapendi.state.CellState` to compute the cell's behaviour:
membrane water balance, gas transport, voltage and thermal sub-models.

Two top-level steady-state models are available:

- :class:`~marapendi.cell.explicit_steady_state.ExplicitSteadyStateModel` —
  MEA temperature is estimated analytically (one forward pass).
- :class:`~marapendi.cell.implicit_steady_state.ImplicitSteadyStateModel` —
  MEA temperature is solved self-consistently with the heat balance via a
  nonlinear root-find.  Warm-start across successive calls is built in.

Both models share the same two-step API::

    model = ExplicitSteadyStateModel()          # or ImplicitSteadyStateModel()
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
and :class:`~marapendi.simulation.conditions.SideConditions` (see also
:doc:`correlations`).

.. automodule:: marapendi.cell.explicit_steady_state
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.cell.implicit_steady_state
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.water_balance
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.transport
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.voltage
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: marapendi.thermal
   :members:
   :undoc-members:
   :show-inheritance:
