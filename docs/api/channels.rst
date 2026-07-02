Flow Channels
=============

:class:`~marapendi.channel.flow_channels.FlowChannel` inherits from
:class:`~marapendi.porous_layers.porous_layers.PorousLayer` so that the channel
participates in the same gas-transport pipeline as the GDL and MPL.  It holds
channel geometry (width, height, length, number of parallel channels) and delegates
gas-transport resistance to a pluggable
:class:`~marapendi.channel.gas_transport_resistance.ChannelGasResistanceModel`.

Two resistance models are available:

- :class:`~marapendi.channel.gas_transport_resistance.ChannelGasResistanceModel` —
  Sherwood-number approach (Kim et al. 2022) combining a diffusion sub-resistance
  and a convection sub-resistance.
- :class:`~marapendi.channel.gas_transport_resistance.BakerChannelGasResistanceModel` —
  empirical correlations from Baker et al. (2009) for the sub-resistances.

Component
---------

.. autoclass:: marapendi.channel.flow_channels.FlowChannel
   :members:
   :show-inheritance:

Transport models
----------------

.. autoclass:: marapendi.channel.gas_transport_resistance.ChannelGasResistanceModel
   :members:
   :show-inheritance:

.. autoclass:: marapendi.channel.gas_transport_resistance.BakerChannelGasResistanceModel
   :members:
   :show-inheritance:
