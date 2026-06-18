.. _how_to_use:

How to use marapendi
====================

marapendi can be used in three ways:

1. **Imported in a script**::

    python my_simulation.py

2. **Imported in another package**:

   .. code-block:: python

        import marapendi as mrpd

3. **Interactively in a Jupyter notebook**::

    jupyter notebook

   Example notebooks are available in the ``notebooks/`` folder.

Quick start
-----------

The shortest path to a polarization curve:

.. code-block:: python

    import numpy as np
    import marapendi as mrpd

    # Build the cell (see "Building a cell model" for details)
    cell = mrpd.FuelCell(...)

    # Define operating conditions
    ca_cond = mrpd.OperatingConditions(
        inlet_temperature=353.15, inlet_pressure=1.5e5, outlet_pressure=1.5e5,
        dry_o2_mole_fraction=0.21, inlet_relative_humidity=0.5, stoichiometry=2.0,
    )
    an_cond = mrpd.OperatingConditions(
        inlet_temperature=353.15, inlet_pressure=1.5e5, outlet_pressure=1.5e5,
        dry_h2_mole_fraction=1.0, inlet_relative_humidity=0.5, stoichiometry=1.5,
    )

    # Sweep current density
    i = np.linspace(1e3, 2e4, 20)   # A/m²
    V = cell.compute_ui_curve(i, 353.15, ca_cond, an_cond)

Parameter estimation
--------------------

Use :class:`~marapendi.estimation.estimation.SteadyStateModel` to fit model
parameters against experimental data:

.. code-block:: python

    def model_fn(params):
        cell = build_cell(i0=params['i0'], alpha=params['alpha'])
        voltages = [cell.compute_ui_curve(...) for i in current_densities]
        return np.array(voltages)

    estimator = mrpd.SteadyStateModel(model_fn, {'i0': 1e-4, 'alpha': 0.5})
    estimator.set_unknown_params([
        ('i0',    (1e-7, 1e-2), True, '$i_0$'),
        ('alpha', (0.2, 0.8),   True, r'$\alpha$'),
    ])
    sol, p_est = estimator.estimate(V_exp, t=0, popsize=15, ftol=1e-9)

See ``notebooks/02_parameter_estimation.ipynb`` for a complete worked example.
