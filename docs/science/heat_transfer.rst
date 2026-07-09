Heat transfer
=================

The release of heat due to the electrochemical reactions and Joule heating in the MEA leads to
an increase of the MEA temperature, which can reach several degrees Celsius, as Thomas
et al. (2013) have shown. 

The steady-state MEA temperature rise above the stack/coolant temperature
:math:`T` follows a lumped heat balance, which assumes a uniform temperature across
the MEA (:meth:`~marapendi.models.thermal.ThermalModel.mea_temperature`):

.. math::

    \dot q = i\left(-\frac{\Delta H_\mathrm{HHV}(T)}{2F} - V_\mathrm{cell}\right),
    \qquad
    T_\mathrm{MEA} = T + \dot q\, R_\mathrm{th},

Lumping the entire MEA into a single temperature is
reasonable because most of that heat — entropic and irreversible reaction
heat, plus Joule heating — is actually produced in the membrane and catalyst
layers (Huang et al., 2022), and their thickness, and hence their thermal
resistance, is small next to that of the GDL and MPL. The temperature drop
between the MEA and the coolant is therefore dominated by the GDL/MPL layers, and the
thermal resistance can be calculated as
(:meth:`~marapendi.models.thermal.ThermalModel.heat_transfer_resistance`):

.. math::

    R_\mathrm{th} =
        \left(\frac{1}{R_\mathrm{th}^\mathrm{ca}}
        + \frac{1}{R_\mathrm{th}^\mathrm{an}}\right)^{-1}

where each side's resistance sums the through-plane resistance of its porous
layers (:attr:`~marapendi.porous_layers.porous_layers.PorousLayer.thermal_resistance`)
and its thermal contact resistance
(:attr:`~marapendi.cell.cell.CellSide.thermal_contact_resistance`)
(:meth:`~marapendi.models.thermal.ThermalModel.side_heat_transfer_resistance`):

.. math::

    R_\mathrm{th}^{ca/an} = 
        \frac{\delta^{ca/an}_\mathrm{GDL}}{k^{ca/an}_\mathrm{GDL}} +
        \frac{\delta^{ca/an}}{\mathrm{MPL}/k^{ca/an}_\mathrm{MPL}} + 
        \mathrm{TCR}^{ca/an}

.. note:: 
    We neglect the impact of liquid water on the thermal conductivities of the
    different layers or on the thermal contact resistance. 

Because :math:`\dot q` itself depends on the cell voltage, solving self-
consistently for :math:`T_\mathrm{MEA}` and :math:`V_\mathrm{cell}` together
means iterating a stiff coupled system.
:meth:`~marapendi.models.thermal.ThermalModel.mea_temperature` can instead
sidestep that coupling by assuming a constant HHV efficiency, replacing the
LHV-based :math:`\dot q` above with :math:`\dot q = i \times 0.7\ \mathrm{V}`. This is
done in the *explicit* model
(:class:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel`), 
while the *implicit* model
(:class:`~marapendi.models.base.implicit_steady_state.ImplicitSteadyStateModel`)
iterates :math:`T_\mathrm{MEA}` and :math:`V_\mathrm{cell}` to
self-consistency using the actual LHV-based heat release above (see :doc:`steady_state_model`). 
The explicit model is faster and vectorises trivially; the implicit model is more accurate
at high current density, where the thermal feedback on water content and
kinetics is significant. See :doc:`/auto_examples/plot_03_implicit_vs_explicit`
for a direct comparison of the two on the same operating conditions.

See :doc:`../user_guide/extending_models`  for how to add an
explicit convective coolant term to :math:`R_\mathrm{th}` by subclassing
:class:`~marapendi.models.thermal.ThermalModel`, :doc:`steady_state_model` for
how :math:`T_\mathrm{MEA}` is obtained analytically vs. self-consistently at
steady state, and :doc:`transient_model` for the time-dependent version of
this same balance.

References
--------------

Thomas, A. et al. *J. Electrochem. Soc.* **160**, F191–F204 (2013).

Huang, Y. et al. *Energy Convers. Manage.* **254**, 115221 (2022).
