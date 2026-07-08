Transient model and load cycles
=====================================

:class:`~marapendi.models.base.transient.TransientModel` extends the
steady-state building blocks from :doc:`kinetics`, :doc:`membrane` and
:doc:`transport` into a coupled ODE system with state vector :math:`x =
[T_\mathrm{MEA},\, \lambda_1, \ldots, \lambda_n]` (MEA temperature followed by
the membrane water content at each of ``n_memb_mesh`` finite-volume nodes).

MEA temperature
--------------------

The same lumped thermal resistance :math:`R_\mathrm{th}` from
:doc:`transport` is integrated in time against the MEA's areal heat capacity
:math:`C_\mathrm{MEA}` = ``cell.mea_surface_heat_capacity`` (J·m⁻²·K⁻¹),
rather than solved for its steady-state value
(:meth:`~marapendi.models.thermal.ThermalModel.temperature_rate_of_change`):

.. math::

    \frac{dT_\mathrm{MEA}}{dt} = \frac{
        \dot q - \Delta T_\mathrm{MEA} / R_\mathrm{th}
    }{C_\mathrm{MEA}},
    \qquad
    \Delta T_\mathrm{MEA} = T_\mathrm{MEA} - T,

with :math:`\dot q` the same heat-release rate as the steady-state model (see
:doc:`transport`) and :math:`T` the instantaneous stack/coolant temperature.

Membrane water-content profile
------------------------------------

Rather than solving the closed-form :math:`\lambda(\xi)` of
:class:`~marapendi.models.water_balance.membrane.MembraneWaterBalanceModel`
(:doc:`membrane`), the transient model prescribes :math:`\lambda` at each mesh
node from the ODE state and only evaluates the local diffusive and
electroosmotic-drag fluxes
(:class:`~marapendi.models.water_balance.membrane_transient.MembraneWaterBalanceTransientModel`):

.. math::

    J_\mathrm{diff}(\xi) = -\frac{1}{R_D}\frac{\partial\lambda}{\partial\xi},
    \qquad
    J_\mathrm{EOD}(\xi) = Pe\cdot\lambda(\xi).

The finite-volume flux divergence then gives the rate of change at each node
(:meth:`~marapendi.models.water_balance.water_balance.WaterBalanceModel.membrane_water_rate_of_change`):

.. math::

    \frac{d\lambda_k}{dt} = -\frac{1}{c_\mathrm{mb}^\mathrm{dry}}\,
        \frac{J_{w,k+1/2} - J_{w,k-1/2}}{\Delta\xi},

with the boundary fluxes at :math:`k=0` and :math:`k=n` taken from the
catalyst-layer absorption/desorption terms of the membrane model — i.e. the
transient membrane model and the steady-state one share the same boundary
physics, only the interior evolution differs. This discretisation follows
Goshtasbi et al., *J. Electrochem. Soc.* **167**, 024518 (2020).

At each time step,
:meth:`~marapendi.models.base.transient.TransientModel.f_transient` re-derives
gas compositions and flow rates from the (possibly time-varying) operating
conditions, recomputes cathode liquid saturation quasi-statically from the net
water flux, and re-evaluates gas transport and cell voltage — so the only
quantities actually integrated in time are :math:`T_\mathrm{MEA}` and
:math:`\lambda`; everything else is quasi-static at each instant.

Handling non-smooth driving conditions
--------------------------------------------

A load cycle's operating conditions are typically piecewise smooth (step
changes in current density, ramps in temperature) rather than globally smooth.
:meth:`~marapendi.simulation.load_cycles.LoadCycle.discontinuity_times`
collects every such kink from the cycle's
:class:`~marapendi.simulation.load_cycles.PiecewiseProfile` fields, and
:meth:`TransientModel.solve` uses them to split
:func:`scipy.integrate.solve_ivp` into one call per smooth sub-interval,
restarting cleanly at each breakpoint — because an adaptive solver's local
error estimate assumes smoothness over the step it just took, and that
assumption is violated exactly at a kink.

Standardised driving cycles
--------------------------------

Two literature load cycles are implemented as
:class:`~marapendi.simulation.load_cycles.LoadCycle` subclasses for
automotive-relevant transient validation:

* **ID-FAST**
  (:class:`~marapendi.simulation.load_cycles.idfast.IDFastCycle`) — the
  Improved Dynamic Fuel-cell ASsessment Test protocol of Colombo et al., *J.
  Power Sources* **553**, 232–250 (2023), derived from a real automotive
  fleet duty dataset: 3925 s total, a 2005 s cold section followed by a
  1920 s hot section, including a short idle-stop period (cathode air flow
  cut, modelled as a very low dry-O₂ mole fraction) where current is drawn
  through an external resistance until the voltage collapses.
* **FC-DLC / NEDC**
  (:class:`~marapendi.simulation.load_cycles.nedc.NEDCCycle`) — the Fuel Cell
  Dynamic Load Cycle standardised by the JRC/FCH-JU, Tsotridis et al., EUR
  27632 EN (2015), Appendix F, derived from the New European Driving Cycle:
  1181 s, 35 piecewise-constant current steps.

See :doc:`../user_guide/load_cycles` for how to build and drive one of these
(or a hand-built cycle) in practice.
