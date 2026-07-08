Gas transport
=================

Porous-layer diffusion
----------------------------

:class:`~marapendi.models.diffusion.PorousGasDiffusionModel` combines
molecular (Fickian) and Knudsen diffusion resistances in series through each
porous layer (GDL, MPL, CL), both penalised by the liquid saturation computed
in :doc:`two_phase_flow`
(:meth:`~marapendi.models.diffusion.PorousGasDiffusionModel.total_diffusion_resistance`):

.. math::

    f(s) = \left[\max(1-s,\,10^{-6})\right]^{m}, \qquad
    L_\mathrm{eff} = \frac{\delta\,\tau}{\varepsilon\, f(s)},

.. math::

    R_\mathrm{diff} = L_\mathrm{eff}
        \left(\frac{1}{D_{ij}} + \frac{1}{D_\mathrm{Kn}}\right),
    \qquad
    D_\mathrm{Kn} = \frac{d_\mathrm{pore}}{3}
        \sqrt{\frac{8RT}{\pi\, M}},

where :math:`m` = ``water_saturation_exponent``, :math:`\delta` the layer
thickness, :math:`\tau` the tortuosity, :math:`d_\mathrm{pore}` the pore
diameter, :math:`M` the species molecular weight, and :math:`D_{ij}` the
binary molecular diffusivity.

End-to-end transport resistance
-------------------------------------

:class:`~marapendi.models.gas_transport_resistance.GasTransportModel` sums the
porous-layer resistances above with the channel resistance from
:doc:`flow_channels`, and — for O₂ only — the ionomer-film resistance from
:doc:`catalyst_layer`, to give the total resistance between the channel and
the reaction site for each species (O₂, H₂, H₂O). It then uses these
resistances to back out the catalyst-layer gas composition from the
channel-side composition and the local consumption/production rates,
supplying the reactant activity :math:`a` used in :doc:`orr_kinetics` and the
vapor concentration used by :doc:`water_balance` at the membrane/CL
interface.
