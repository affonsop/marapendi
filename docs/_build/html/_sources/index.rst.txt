marapendi
=========

**marapendi** is a Python framework for physics-based modelling of proton-exchange
membrane (PEM) and anion-exchange membrane (AEM) electrochemical cells. The current
release targets PEM fuel cells; AEM electrolyzer support is planned for a future version.

Philosophy
----------

**marapendi** separates the *description* of a cell from the *calculations*
performed on it:

``components`` (:mod:`marapendi.cell`, :mod:`marapendi.porous_layers`, :mod:`marapendi.membrane`, :mod:`marapendi.channel`)
    Dataclasses holding the static, measurable properties of a cell's
    components (geometry, porosity, permeability, ionomer parameters,
    catalyst loading, ...), together with the correlation models that turn
    these properties into transport and electrochemical quantities. A
    :class:`~marapendi.cell.fuelcell.FuelCell` assembles a cathode and anode
    :class:`~marapendi.cell.fuelcell.FuelCellSide` (each with a catalyst layer,
    gas diffusion layer, optional microporous layer and flow channel) and a
    :class:`~marapendi.membrane.membrane_base.Membrane`.

``state`` (:mod:`marapendi.cell.state`)
    Dataclasses mirroring the shape of :class:`~marapendi.cell.fuelcell.FuelCell`
    but holding the *physical variables* (temperature, pressure, gas composition,
    saturation, water content, fluxes, ...) at one operating point. State objects
    are pure data: no physics lives here.

``models`` (:mod:`marapendi.cell.explicit_steady_state`, :mod:`marapendi.cell.implicit_steady_state`, :mod:`marapendi.cell.transient`, :mod:`marapendi.water_balance`)
    Orchestration classes that combine a :class:`~marapendi.cell.fuelcell.FuelCell`
    and a :class:`~marapendi.cell.state.CellState` to compute the cell's behaviour —
    membrane water balance, gas transport, voltage and thermal sub-models.
    :class:`~marapendi.cell.explicit_steady_state.ExplicitSteadyStateModel` and
    :class:`~marapendi.cell.implicit_steady_state.ImplicitSteadyStateModel` provide
    steady-state evaluation; :class:`~marapendi.cell.transient.TransientModel`
    integrates the coupled MEA-temperature and water-profile ODEs.

``correlations`` (:mod:`marapendi.thermo`, :mod:`marapendi.simulation`)
    Stateless physics building blocks (water thermodynamics, electrochemical
    kinetics, gas mixture properties, operating conditions) used by the
    components and models above.

``estimation`` (:mod:`marapendi.estimation`)
    :class:`~marapendi.estimation.polarization_curve_calibration.SteadyStatePolarizationCurveCalibration`
    fits kinetic and transport parameters to multi-condition polarization and HFR data
    using differential-evolution global optimisation, with k-fold cross-validation and
    automatic model-complexity selection via the 1-SE rule.

The steady-state model and the underlying correlations implement the model
described in Affonso Nobrega et al., *J. Electrochem. Soc.* **173**, 114503
(2026), to which this documentation defers for the physical and mathematical
description of the cell model.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   auto_examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
