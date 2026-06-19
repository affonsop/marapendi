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

Both models are accessible via :meth:`~marapendi.cell.FuelCell.compute_ui_curve`
using the ``model`` keyword argument.

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
