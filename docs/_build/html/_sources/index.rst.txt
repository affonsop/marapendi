marapendi
=========

**marapendi** is a Python framework for modelling proton and anion exchange
membrane (PEM/AEM) electrochemical cell devices, including fuel cells and
water electrolyzers.

Philosophy
----------

**marapendi** separates the *description* of a cell from the *calculations*
performed on it:

``components`` (:mod:`marapendi.cell`, :mod:`marapendi.catalyst_layers`,
:mod:`marapendi.porous_layers`, :mod:`marapendi.membrane`,
:mod:`marapendi.flow_channels`, :mod:`marapendi.ionomer`)
    Dataclasses holding the static, measurable properties of a cell's
    components (geometry, porosity, permeability, ionomer parameters,
    catalyst loading, ...), together with the correlation models that turn
    these properties into transport and electrochemical quantities. A
    :class:`~marapendi.cell.Cell` assembles a cathode and anode
    :class:`~marapendi.cell.CellSide` (each with a catalyst layer, gas
    diffusion layer, optional microporous layer and flow channel) and a
    :class:`~marapendi.membrane.Membrane`.

``state`` (:mod:`marapendi.state`)
    Dataclasses mirroring the shape of :class:`~marapendi.cell.Cell` but
    holding the *physical variables* (temperature, pressure, gas
    composition, saturation, water content, fluxes, ...) at one operating
    point. State objects are pure data: no physics lives here.

``models`` (:mod:`marapendi.model`, :mod:`marapendi.water_balance`,
:mod:`marapendi.transport`, :mod:`marapendi.voltage`,
:mod:`marapendi.thermal`)
    Orchestration classes that combine a :class:`~marapendi.cell.Cell` and a
    :class:`~marapendi.state.CellState` to compute the cell's behaviour —
    membrane water balance, gas transport, voltage and thermal sub-models.
    :class:`~marapendi.model.ExplicitSteadyStateModel` composes these into a
    single explicit steady-state evaluation.

``correlations`` (:mod:`marapendi.water`, :mod:`marapendi.electrochemistry`,
:mod:`marapendi.membrane_permeation_models`,
:mod:`marapendi.transport_models`, :mod:`marapendi.gas`,
:mod:`marapendi.conditions`, :mod:`marapendi.constants`)
    Stateless physics building blocks (water thermodynamics, electrochemical
    kinetics, membrane permeation and transport correlations, gas mixture
    properties, operating conditions, and physical constants) used by the
    components and models above.

``estimation`` (:mod:`marapendi.estimation.parameter_estimation`,
:mod:`marapendi.estimation.cross_validation`)
    Parameter estimation and (leave-one-out) cross-validation routines for
    fitting the steady-state model to experimental polarization data.

The steady-state model and the underlying correlations implement the model
described in Affonso Nobrega et al., *J. Electrochem. Soc.* 173, 114503
(2026), to which this documentation defers for the physical and mathematical
description of the cell model.

Experimental: the ``dynamic`` package
--------------------------------------

:mod:`marapendi.dynamic` is a separate, transient-capable implementation
inspired by Yang et al. (2019). It is under evaluation and not yet part of
the documented public API.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/components
   api/models
   api/correlations
   api/estimation
   api/tools

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
