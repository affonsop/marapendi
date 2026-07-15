marapendi
=========

**marapendi** is a Python framework for physics-based modelling of proton-exchange
membrane (PEM) and anion-exchange membrane (AEM) electrochemical cells. The current
release targets PEM fuel cells; AEM electrolyzer support is planned for a future version.

**marapendi** offers:

- The basis for the implementation of 0D (and up to 1D) physics-based models of
  PEM/AEM fuel cells and electrolyzers.
- Very fast steady-state and transient 0D cell models — low enough computational
  cost to make sensitivity analysis, parameter estimation and cross-validation
  practical (see :doc:`science/parameter_estimation`).
- An easy-to-use API for defining, calibrating and simulating cell models in a
  few lines of code (see :doc:`installation` for a runnable example).
- Pre-defined correlations and sub-models for heat transfer, reaction kinetics,
  two-phase transport, membrane water balance, ohmic losses, and more (see
  :doc:`science/index`).
- Straightforward extension to new models: sub-models are ordinary Python
  classes, so overriding one correlation or swapping a whole sub-model is a
  matter of subclassing (see :doc:`user_guide/extending_models`).
- Multiple runnable examples demonstrating **marapendi**'s capabilities (see
  :doc:`auto_examples/index`).
- Detailed documentation and openly readable, commented code, designed for
  transparency and easy understanding.
- A transient 0D model available as a MATLAB/Simulink S-function block (see
  :doc:`user_guide/simulink_block`).

Philosophy
----------

**marapendi** is structured and written in a way to make the implementation of models and sub-models, 
and their use for simulation and parameter estimation easy. 

**marapendi** separates the *description* of a cell from the *calculations*
performed on it:

``components`` (:mod:`marapendi.cell`, :mod:`marapendi.porous_layers`, :mod:`marapendi.membrane`, :mod:`marapendi.channel`, :mod:`marapendi.electrolyte`)
    Dataclasses holding the static, measurable properties of a cell's
    components (geometry, porosity, permeability, ionomer parameters,
    catalyst loading, ...), together with the correlation models that turn
    these properties into transport and electrochemical quantities. A
    :class:`~marapendi.cell.fuelcell.FuelCell` assembles a cathode and anode
    :class:`~marapendi.cell.fuelcell.FuelCellSide` (each with a catalyst layer,
    gas diffusion layer, optional microporous layer and flow channel) and a
    :class:`~marapendi.membrane.membrane_base.Membrane`.

``state`` (:mod:`marapendi.simulation.state`)
    Dataclasses mirroring the shape of :class:`~marapendi.cell.fuelcell.FuelCell`
    but holding the *physical variables* (temperature, pressure, gas composition,
    saturation, water content, fluxes, ...) at one operating point. State objects
    are pure data: no physics lives here.

``models`` (:mod:`marapendi.models`)
    Orchestration classes that combine a :class:`~marapendi.cell.fuelcell.FuelCell`
    and a :class:`~marapendi.simulation.state.CellState` to compute the cell's
    behaviour. ``models.base`` holds the top-level solvers —
    :class:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel` and
    :class:`~marapendi.models.base.implicit_steady_state.ImplicitSteadyStateModel`
    for steady-state evaluation, :class:`~marapendi.models.base.transient.TransientModel`
    for the coupled MEA-temperature/water-profile ODEs — while
    :mod:`~marapendi.models.thermal`, :mod:`~marapendi.models.voltage`,
    :mod:`~marapendi.models.gas_transport_resistance`, ``models.water_balance``,
    :mod:`~marapendi.models.darcy` and :mod:`~marapendi.models.diffusion` provide the
    thermal, voltage, gas-transport, membrane water-balance and two-phase-transport
    sub-models each solver is built from.

``correlations`` (:mod:`marapendi.models.thermo`, :mod:`marapendi.simulation`)
    Stateless physics building blocks (water thermodynamics, electrochemical
    kinetics, gas mixture properties, operating conditions) used by the
    components and models above. :mod:`marapendi.simulation` also holds
    :class:`~marapendi.simulation.load_cycles.LoadCycle` and the standardised
    ID-FAST/FC-DLC driving cycles used to drive the transient model in time.

``estimation`` (:mod:`marapendi.estimation`)
    :class:`~marapendi.estimation.polarization_curve_calibration.SteadyStatePolarizationCurveCalibration`
    fits kinetic and transport parameters to multi-condition polarization and HFR data
    using differential-evolution global optimisation, with k-fold cross-validation and
    automatic model-complexity selection via the 1-SE rule.

The steady-state model and the underlying correlations implement the model
described in Affonso Nobrega et al., *J. Electrochem. Soc.* **173**, 114503
(2026), to which this documentation defers for the physical and mathematical
description of the cell model.

``interop`` (:mod:`marapendi.interop`)
    A thin adapter (:mod:`marapendi.interop.simulink_bridge`) exposing
    :class:`~marapendi.models.base.transient.TransientModel` to MATLAB through
    plain scalars, lists, and dicts. It backs the ``TransientPEMFC`` Simulink
    block (``matlab/transient_pemfc/``, see :doc:`user_guide/simulink_block`),
    which calls the live Python model at every solver step — no physics is
    re-implemented in MATLAB, and cell parameters can be supplied either as a
    dotted Python builder path or as a plain MATLAB struct.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Science

   science/index

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
