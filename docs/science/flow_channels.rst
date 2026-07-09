Flow channels
=================

Species transport from the flow channel to the porous-layer stack combines a
diffusive and a convective sub-resistance. The default Sherwood-number
approach
(:class:`~marapendi.channel.gas_transport_resistance.ChannelGasResistanceModel`)
gives

.. math::

    R_\mathrm{diff}^\mathrm{ch} = \frac{d_h}{Sh\, D},
    \qquad
    R_\mathrm{conv}^\mathrm{ch} = \frac{B_\mathrm{ch}\, L\, W}{2}
        \left(1 + \frac{1}{\mathrm{ch/land}}\right)\frac{n_\mathrm{ch}}{\dot V},

with :math:`d_h` the channel hydraulic diameter, :math:`Sh` the Sherwood
number, :math:`D` the binary diffusion coefficient, :math:`L, W` the channel
length/width, ``channel_land_ratio`` the channel-to-land width ratio,
:math:`n_\mathrm{ch}` the number of parallel channels, and :math:`\dot V` the
volumetric flow rate.

An alternative empirical form follows Baker et al. (2009)
(:class:`~marapendi.channel.gas_transport_resistance.BakerChannelGasResistanceModel`),
replacing :math:`d_h/Sh` with :math:`A_\mathrm{ch}\times(\text{channel
half-width})` and the convective term with a length/half-width correlation
calibrated against their flow-field data.

The channel resistance :math:`R_\mathrm{diff}^\mathrm{ch} +
R_\mathrm{conv}^\mathrm{ch}` is one term in the total end-to-end species
transport resistance assembled in :doc:`gas_transport` — in series with the
porous-layer resistances, and (for O₂) the ionomer-film resistance from
:doc:`catalyst_layer`.

References
--------------

Baker, D. R. et al. *J. Electrochem. Soc.* **156**, B991 (2009).
