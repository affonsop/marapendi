The science behind marapendi
================================

The :doc:`../user_guide/index` shows *how* to call **marapendi**. This section
explains *what* each model actually computes: the governing equations, in the
same symbols used in the code, and the paper each correlation or numerical
scheme comes from. The section order follows the model description in
Affonso Nobrega et al., *J. Electrochem. Soc.* **173**, 114503 (2026).

Where **marapendi**'s own contribution is a specific correlation or model
(rather than an implementation of prior literature), the page says so
explicitly and cites that paper — it is the one the steady-state voltage and
membrane water-balance model were developed for and validated against.
Variable names below follow the code (which follows that paper's notation):
current density :math:`i`, cell voltage :math:`V_\mathrm{cell}`, reversible
voltage :math:`E_\mathrm{rev}`, activation overpotential
:math:`\eta_\mathrm{act}`, ohmic overpotential :math:`\eta_\mathrm{ohm}`,
membrane water content :math:`\lambda`, and the non-dimensional Péclet/Biot
numbers :math:`Pe`, :math:`Bi`.

.. note::

   This section documents the physics currently implemented for **PEM fuel
   cells**; AEM electrolyzer support (see :mod:`marapendi.membrane.aem`,
   :mod:`marapendi.electrolyte.koh`) is under active development — see
   :doc:`additional_models` for what is available so far.

.. toctree::
   :maxdepth: 1

   cell_voltage
   heat_transfer
   two_phase_flow
   membrane_correlations
   water_balance
   catalyst_layer
   orr_kinetics
   flow_channels
   gas_transport
   steady_state_model
   transient_model
   parameter_estimation
   degradation
   additional_models
