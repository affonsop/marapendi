Science
================================

This section explains the different models, the governing equations and
correlations used in **marapendi**. Most of them are described in
Affonso Nobrega et al. (2026), although some models were modified with
respect to the version in this paper.

.. note::

   This section documents the physics currently implemented for **PEM fuel
   cells**; AEM electrolyzer support (see :mod:`marapendi.membrane.aem`,
   :mod:`marapendi.electrolyte.koh`) is under active development — see
   :doc:`additional_models` for what is available so far.

.. toctree::
   :maxdepth: 1

   cell_voltage
   heat_transfer
   water
   water_balance
   two_phase_flow
   membrane_correlations
   catalyst_layer
   orr_kinetics
   flow_channels
   gas_transport
   steady_state_model
   transient_model
   parameter_estimation
   degradation
   additional_models

References
--------------

Affonso Nobrega, P. et al. *J. Electrochem. Soc.* **173**, 114503 (2026).
