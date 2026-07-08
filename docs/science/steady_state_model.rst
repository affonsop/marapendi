Steady-state model
=======================

Both steady-state models run the same solve sequence — heat transfer
(:doc:`heat_transfer`) → water balance (:doc:`water_balance`, using
:doc:`membrane_correlations`) → gas transport (:doc:`gas_transport`,
:doc:`flow_channels`) → cell voltage (:doc:`cell_voltage`, :doc:`orr_kinetics`,
:doc:`catalyst_layer`) — and differ only in how the MEA temperature
:math:`T_\mathrm{MEA}` is obtained, since :math:`T_\mathrm{MEA}` and
:math:`V_\mathrm{cell}` are coupled: the water balance and kinetics both
depend on :math:`T_\mathrm{MEA}`, while :math:`T_\mathrm{MEA}` itself depends
on the heat released, which depends on :math:`V_\mathrm{cell}` (see
:doc:`heat_transfer`).

Explicit model
-------------------

:class:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel`
breaks that coupling with a single forward pass: :math:`T_\mathrm{MEA}` is
estimated once from a fixed-efficiency approximation (0.7 V) *before* solving
for the actual :math:`V_\mathrm{cell}`, rather than iterating the two to
self-consistency. This is fast and vectorises trivially over an array of
operating points, at the cost of some accuracy in :math:`T_\mathrm{MEA}` at
high current density (where the actual voltage departs further from 0.7 V).

Implicit model
-------------------

:class:`~marapendi.models.base.implicit_steady_state.ImplicitSteadyStateModel`
extends the explicit model by iterating :math:`T_\mathrm{MEA}` and
:math:`V_\mathrm{cell}` to self-consistency: at each iterate of
:math:`V_\mathrm{cell}`, :math:`T_\mathrm{MEA}` is recomputed from the actual
heat release (:doc:`heat_transfer`), the water balance and voltage are
re-solved at that temperature, and the residual :math:`V_\mathrm{cell}^\mathrm{guess}
- V_\mathrm{cell}^\mathrm{solved}` is driven to zero. Because each
current-density point is independent (no spatial coupling), this residual is
diagonal across points, so :func:`scipy.optimize.newton` (elementwise secant
method) solves the whole array in one vectorised call without building a
dense Jacobian — warm-started from the explicit model's 0.7 V estimate. The
implicit model is more accurate at high current density, where the thermal
feedback on water content and kinetics is significant; the explicit model is
faster and normally sufficient for :doc:`parameter_estimation`.

See :doc:`../user_guide/polarization_curve` for the two-line usage pattern
shared by both models, and :doc:`transient_model` for the time-dependent
extension of the same physics.
