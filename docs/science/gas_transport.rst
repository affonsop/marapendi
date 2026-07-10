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

    f(s) = \left[\max(1-s,\,10^{-6})\right]^{n_s}, \qquad
    L_\mathrm{eff} = \frac{\delta\,\tau}{\varepsilon\, f(s)},

.. math::

    R_\mathrm{diff} = L_\mathrm{eff}
        \left(\frac{1}{D_{ij}} + \frac{1}{D_\mathrm{Kn}}\right),
    \qquad
    D_\mathrm{Kn} = \frac{d_\mathrm{pore}}{3}
        \sqrt{\frac{8RT}{\pi\, M}},

where :math:`n_s` = ``water_saturation_exponent``, :math:`\delta` the layer
thickness, :math:`\tau` the tortuosity, :math:`d_\mathrm{pore}` the pore
diameter, :math:`M` the species molecular weight, and :math:`D_{ij}` the
binary molecular diffusivity.

End-to-end transport resistance
-------------------------------------

:class:`~marapendi.models.gas_transport_resistance.GasTransportModel`
(:meth:`~marapendi.models.gas_transport_resistance.GasTransportModel.gas_transport_resistance`)
sums the porous-layer resistances above with the channel resistance from
:doc:`flow_channels`, and — for O₂ only — the ionomer-film resistance from
:doc:`catalyst_layer`, to give the total resistance :math:`R_\mathrm{tot}`
between the channel and the reaction site for each species (O₂, H₂, H₂O):

.. math::

    R_\mathrm{tot} = \sum_{k \,\in\, \mathrm{GDL,\,MPL,\,CL}} R_{\mathrm{diff},k}
        + R_\mathrm{diff}^\mathrm{ch} + R_\mathrm{conv}^\mathrm{ch}
        + \delta_{\mathrm{O_2}}\,R_\mathrm{ion},

where the last (ionomer-film) term is only added for O₂
(:doc:`catalyst_layer`). Treating each species as a resistance network
carrying its local consumption/production flux, the catalyst-layer
concentration follows directly from the channel-side concentration
(:meth:`~marapendi.models.gas_transport_resistance.GasTransportModel.calculate_gas_concentrations`):

.. math::

    c_\mathrm{CL,reactant} = c_\mathrm{CH,reactant}
        - J_\mathrm{reactant}\, R_\mathrm{tot}, \qquad
    c_{\mathrm{CL},v} = c_{\mathrm{CH},v}
        + J_v\, R_{\mathrm{tot},v},

with :math:`J` the local molar flux (consumption, taken positive for
the reactant; production, for water vapor). The nitrogen concentration 
is calculated by ensuring the gas pressure remains constant: 

.. math:: 

    x_\mathrm{N_2} = 1 - x_\mathrm{H_2} - x_\mathrm{O_2} - x_v