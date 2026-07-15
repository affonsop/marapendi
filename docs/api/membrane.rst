Membrane and Ionomer
====================

The membrane subsystem has two separate class hierarchies that are composed
rather than merged:

**Ionomer** — transport correlations for the polymer electrolyte phase, used by
both the membrane and the catalyst layers.
:class:`~marapendi.components.membrane.ionomer_base.Ionomer` is the abstract base;
:class:`~marapendi.components.membrane.pem.PFSAIonomer` provides empirical fits for Nafion-type
ionomers (proton conductivity, O₂ permeability, H₂ permeability, electroosmotic
drag, and water diffusivity).

**Membrane** — geometric properties (dry thickness) and water-balance correlations at
the membrane level.  :class:`~marapendi.components.membrane.membrane_base.Membrane` delegates
transport correlations to its composed
:attr:`~marapendi.components.membrane.membrane_base.Membrane.ionomer`.
:class:`~marapendi.components.membrane.pem.PFSA` specialises to PFSA membranes and adds the
Springer et al. (1991) vapor-equilibrium isotherm and liquid equilibrium.

**Water balance models** — solve the 1-D water-content profile across the membrane.
Two implementations are provided:

- :class:`~marapendi.models.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise`
  (default) — boundary conditions derived from a piecewise-linear regression of the
  equilibrium isotherm RH(λ) stored on the ionomer.
- :class:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel` — boundary
  conditions derived from the first-order linear expansion of the isotherm used in
  Affonso Nobrega et al. (2026).

Both are orchestrated by
:class:`~marapendi.models.water_balance.water_balance.WaterBalanceModel`, which also handles
gas-transport resistances and liquid saturation.

Ionomer
-------

.. autoclass:: marapendi.components.membrane.ionomer_base.Ionomer
   :members:
   :show-inheritance:

.. autoclass:: marapendi.components.membrane.pem.PFSAIonomer
   :members:
   :show-inheritance:

Membrane
--------

.. autoclass:: marapendi.components.membrane.membrane_base.Membrane
   :members:
   :show-inheritance:

.. autoclass:: marapendi.components.membrane.pem.PFSA
   :members:
   :show-inheritance:

Water balance models
--------------------

.. autoclass:: marapendi.models.water_balance.water_balance.WaterBalanceModel
   :members:
   :show-inheritance:

.. autoclass:: marapendi.models.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise
   :members:
   :show-inheritance:

.. autoclass:: marapendi.models.water_balance.membrane.MembraneWaterBalanceModel
   :members:
   :show-inheritance:
