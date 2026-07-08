The science behind marapendi
================================

The :doc:`../user_guide/index` shows *how* to call **marapendi**. This section
explains *what* each model actually computes: the governing equations, in the
same symbols used in the code, and the paper each correlation or numerical
scheme comes from.

Where **marapendi**'s own contribution is a specific correlation or model
(rather than an implementation of prior literature), the page says so
explicitly and cites Affonso Nobrega et al., *J. Electrochem. Soc.* **173**,
114503 (2026) — the paper the steady-state voltage and membrane water-balance
model were developed for and validated against. Variable names below follow
the code (which follows that paper's notation): current density :math:`i`,
cell voltage :math:`V_\mathrm{cell}`, reversible voltage :math:`E_\mathrm{rev}`,
activation overpotential :math:`\eta_\mathrm{act}`, ohmic overpotential
:math:`\eta_\mathrm{ohm}`, membrane water content :math:`\lambda`, and the
non-dimensional Péclet/Biot numbers :math:`Pe`, :math:`Bi`.

.. note::

   This section documents the physics currently implemented for **PEM fuel
   cells**; AEM electrolyzer support (see :mod:`marapendi.membrane.aem`,
   :mod:`marapendi.electrolyte.koh`) is under active development — see
   :doc:`additional_models` for what is available so far.

.. toctree::
   :maxdepth: 1

   kinetics
   membrane
   transport
   transient
   estimation
   additional_models

Reading order
-----------------

The pages are organised the way the steady-state solve pipeline itself is
built, so reading top to bottom follows one call to
:meth:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel.solve`:

#. :doc:`kinetics` — reversible voltage and Butler–Volmer/Tafel kinetics
   (:math:`E_\mathrm{rev}`, :math:`\eta_\mathrm{act}`), and catalyst-layer
   charge/species transport.
#. :doc:`membrane` — the analytical membrane water-balance solution that gives
   :math:`\lambda(\xi)` and the ohmic overpotential :math:`\eta_\mathrm{ohm}`.
#. :doc:`transport` — two-phase (liquid water) transport in the porous layers,
   gas-phase diffusion, and the lumped MEA thermal model.
#. :doc:`transient` — the time-dependent extension of the above (MEA
   temperature and membrane water-content ODEs) and the load cycles used to
   drive it.
#. :doc:`estimation` — how unknown kinetic/transport parameters are fit to
   data and how model complexity is selected.
#. :doc:`additional_models` — degradation kinetics and the AEM/KOH
   correlations in preparation for electrolyzer support.
