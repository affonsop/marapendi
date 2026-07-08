Two-phase flow
==================

:class:`~marapendi.models.darcy.DarcyTransportModel` treats liquid water as
the non-wetting phase driven by capillary pressure in the porous layers (GDL,
MPL, CL). The entry (breakthrough) capillary pressure of a layer is set by its
geometry
(:meth:`~marapendi.porous_layers.porous_layers.PorousLayer._compute_breakthrough_pressure`):

.. math::

    P_\mathrm{bt} = \frac{\gamma(T)\,|\cos\theta|}{\sqrt{K/\varepsilon}},

with :math:`\gamma(T)` the water surface tension, :math:`\theta` the layer's
contact angle, :math:`K` the absolute permeability and :math:`\varepsilon` the
porosity. Saturation and capillary pressure are related by a power law with
exponent :math:`n` = ``J_function_exponent``
(:meth:`~marapendi.models.darcy.DarcyTransportModel.capillary_pressure_from_saturation`):

.. math::

    P_c(s) = P_\mathrm{bt}\, s^{\,n}.

Given the non-wetting (liquid) flux :math:`q` through the layer, the
saturation rise across the layer is obtained from a Darcy-flow resistance
:math:`R_s` (``saturation_flow_resistance``, from permeability and relative
permeability exponent :math:`q_\mathrm{rel}` =
``relative_permeability_exponent``) via
(:meth:`~marapendi.models.darcy.DarcyTransportModel.calculate_non_wetting_saturation`):

.. math::

    \Delta s = \left(R_s\, q\, \frac{q_\mathrm{rel}+n}{n}\right)^{1/(q_\mathrm{rel}+n)},
    \qquad
    s_\mathrm{avg} = s_\mathrm{up} + \Delta s\,\frac{q_\mathrm{rel}+n}{q_\mathrm{rel}+n+1}.

The resulting saturation feeds directly into :doc:`gas_transport` (diffusivity
suppression) and :doc:`catalyst_layer` (electrolyte-phase resistance in
alkaline layers).
