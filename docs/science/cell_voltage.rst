Cell voltage
================

The cell voltage assembled by :class:`~marapendi.models.voltage.VoltageModel`
is exactly

.. math::

    V_\mathrm{cell} = E_\mathrm{rev} - \eta_\mathrm{act} - \eta_\mathrm{ohm},

with no additional fitted terms — every loss mechanism the model accounts for
is folded into either the activation overpotential :math:`\eta_\mathrm{act}`
(see :doc:`orr_kinetics`) or the ohmic overpotential :math:`\eta_\mathrm{ohm}`
(membrane proton resistance from :doc:`water_balance` plus catalyst-layer
charge-transport resistance from :doc:`catalyst_layer`).

Reversible (Nernst) voltage
--------------------------------

:func:`~marapendi.models.thermo.electrochemistry.calculate_reversible_cell_voltage`
computes :math:`E_\mathrm{rev}` from the standard Gibbs energy and entropy of
formation of liquid water:

.. math::

    E_\mathrm{rev}(T, Q) = \frac{
        -\Delta G^\circ_{f,\mathrm{H_2O(l)}}
        + \Delta S^\circ_{f,\mathrm{H_2O(l)}} \,(T - T^\circ)
        + R T \ln Q
    }{2F},

where :math:`Q = a_{\mathrm{H_2}} \, a_{\mathrm{O_2}}^{1/2} / a_{\mathrm{H_2O}}`
is the reaction quotient (``activities_ratio`` in the code, built from partial
pressures normalised by a reference pressure), :math:`F` is the Faraday
constant, :math:`R` the gas constant, and :math:`T^\circ` the standard-state
temperature.

Voltage vs. heat release
-----------------------------

The gap between the LHV-equivalent (thermoneutral) voltage and the actual
cell voltage is exactly the heat released at the MEA — see :doc:`heat_transfer`
for how that heat release drives the MEA temperature, and :doc:`steady_state_model`
for how :math:`V_\mathrm{cell}` and :math:`T_\mathrm{MEA}` are solved together
(or one after the other, depending on the model).
