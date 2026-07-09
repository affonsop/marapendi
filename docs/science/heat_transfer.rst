Heat transfer
=================

The MEA sits between two parallel thermal paths (cathode and anode stacks of
porous layers plus their contact resistances)
(:meth:`~marapendi.models.thermal.ThermalModel.heat_transfer_resistance`):

.. math::

    R_\mathrm{th}^{-1} = \sum_{s\,\in\,\{\mathrm{ca,an}\}}
        \left(\sum_{\text{layer} \neq \mathrm{cl}} R_\mathrm{th}^\mathrm{layer}
        + R_\mathrm{contact}\right)^{-1}_s.

The steady-state MEA temperature rise above the stack/coolant temperature
:math:`T` follows a lumped heat balance
(:meth:`~marapendi.models.thermal.ThermalModel.mea_temperature`):

.. math::

    \dot q = i\left(-\frac{\Delta H_\mathrm{LHV}(T)}{2F} - V_\mathrm{cell}\right),
    \qquad
    T_\mathrm{MEA} = T + \dot q\, R_\mathrm{th},

i.e. every irreversible loss (the gap between the LHV-equivalent voltage and
the actual cell voltage — see :doc:`cell_voltage`) is assumed to appear as
heat at the MEA. This lumped treatment follows Ferrara et al. (2018).

See :doc:`../user_guide/extending_models` (Pattern 4) for how to add an
explicit convective coolant term to :math:`R_\mathrm{th}` by subclassing
:class:`~marapendi.models.thermal.ThermalModel`, :doc:`steady_state_model` for
how :math:`T_\mathrm{MEA}` is obtained analytically vs. self-consistently at
steady state, and :doc:`transient_model` for the time-dependent version of
this same balance.

References
--------------

Ferrara, A. et al. *J. Power Sources* **390**, 197–207 (2018).
