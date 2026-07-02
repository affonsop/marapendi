Operating Conditions
====================

:class:`~marapendi.simulation.conditions.CellConditions` bundles the complete set
of operating conditions for one simulation point: current density, stack temperature,
and one :class:`~marapendi.simulation.conditions.SideConditions` per electrode.  It
is the main input to
:meth:`~marapendi.cell.explicit_steady_state.ExplicitSteadyStateModel.set_initial_conditions`
and :meth:`~marapendi.cell.explicit_steady_state.ExplicitSteadyStateModel.solve`::

    conditions = CellConditions(
        current_density=np.linspace(1e3, 2e4, 20),   # A/m² — array for vectorised eval
        cell_temperature=353.15,                       # K
        ca=SideConditions(
            inlet_temperature=353.15, outlet_pressure=1.5e5,
            dry_o2_mole_fraction=0.21, inlet_relative_humidity=0.5, stoichiometry=2.0,
        ),
        an=SideConditions(
            inlet_temperature=353.15, outlet_pressure=1.5e5,
            dry_h2_mole_fraction=1.0, inlet_relative_humidity=0.5, stoichiometry=1.5,
        ),
    )

All fields in :class:`~marapendi.simulation.conditions.CellConditions` and
:class:`~marapendi.simulation.conditions.SideConditions` that carry physical units
accept numpy arrays of the same shape as ``current_density``, so a single ``solve``
call evaluates the entire array of operating points in one vectorised pass.

.. autoclass:: marapendi.simulation.conditions.CellConditions
   :members:
   :show-inheritance:

.. autoclass:: marapendi.simulation.conditions.SideConditions
   :members:
   :show-inheritance:
