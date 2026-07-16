Flow channels
=================

Species transport from the flow channel to the porous-layer stack combines a
diffusive and a convective sub-resistance. The default Sherwood-number
approach, following Kim et al. (2022)
(:class:`~marapendi.models.channel.ChannelGasResistanceModel`,
:meth:`~marapendi.models.channel.ChannelGasResistanceModel.total_resistance`),
gives

.. math::

    R_\mathrm{diff}^\mathrm{ch} = \frac{d_h}{Sh\, D},
    \qquad
    R_\mathrm{conv}^\mathrm{ch} = \frac{B_\mathrm{ch}\, L\, W}{2}
        \left(1 + \frac{1}{\mathrm{ch/land}}\right)\frac{n_\mathrm{ch}}{\dot V},

with :math:`d_h` the channel hydraulic diameter
(:attr:`~marapendi.components.channel.flow_channels.FlowChannel.hydraulic_diameter`),
:math:`Sh` = :attr:`~marapendi.models.channel.ChannelGasResistanceModel.sherwood`,
:math:`D` the binary diffusion coefficient, :math:`L, W` the channel
length/width, ``ch/land`` = :attr:`~marapendi.components.channel.flow_channels.FlowChannel.channel_land_ratio`
the channel-to-land width ratio, :math:`n_\mathrm{ch}` =
:attr:`~marapendi.components.channel.flow_channels.FlowChannel.n_parallel` the number of
parallel channels, :math:`B_\mathrm{ch}` a fitted convective pre-factor
(:attr:`~marapendi.models.channel.ChannelGasResistanceModel.B_ch`),
and :math:`\dot V` the volumetric flow rate.

An alternative empirical form follows Baker et al. (2009)
(:class:`~marapendi.models.channel.BakerChannelGasResistanceModel`),
replacing :math:`d_h/Sh` with
:math:`A_\mathrm{ch}\times(\text{channel half-width})`
(:attr:`~marapendi.components.channel.flow_channels.FlowChannel.half_width`,
:attr:`~marapendi.models.channel.BakerChannelGasResistanceModel.A_ch`)
and the convective term with a length/half-width correlation

.. math::

    R_\mathrm{diff}^\mathrm{ch} = \frac{A_\mathrm{ch}\,(W/2)}{D},
    \qquad
    R_\mathrm{conv}^\mathrm{ch} = \frac{B_\mathrm{ch}\, L}{W/2}\,
        \frac{A_\mathrm{ch,tot}}{\dot V},

with :math:`A_\mathrm{ch,tot}` the total flow cross-section over all parallel
channels
(:attr:`~marapendi.components.channel.flow_channels.FlowChannel.total_flow_section`),
calibrated against Baker et al.'s flow-field data.

The channel resistance :math:`R_\mathrm{diff}^\mathrm{ch} +
R_\mathrm{conv}^\mathrm{ch}`
(:meth:`~marapendi.models.channel.ChannelGasResistanceModel.gas_transport_resistance`)
is one term in the total end-to-end species transport resistance assembled
in :doc:`gas_transport` — in series with the porous-layer resistances, and
(for O₂) the ionomer-film resistance from :doc:`catalyst_layer`.

References
--------------

Kim, H. et al. *Int. J. Heat Mass Transf.* **183**, 122106 (2022).

Baker, D. R. et al. *J. Electrochem. Soc.* **156**, B991 (2009).
