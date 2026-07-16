Flow Channels
=============

:class:`~marapendi.components.channel.flow_channels.FlowChannel` inherits from
:class:`~marapendi.components.porous_layers.porous_layers.PorousLayer` so that the channel
participates in the same gas-transport pipeline as the GDL and MPL.  It holds
channel geometry (width, height, length, number of parallel channels) and delegates
gas-transport resistance to a pluggable
:class:`~marapendi.models.channel.ChannelGasResistanceModel`.

Two resistance models are available:

- :class:`~marapendi.models.channel.ChannelGasResistanceModel` —
  Sherwood-number approach (Kim et al. 2022) combining a diffusion sub-resistance
  and a convection sub-resistance.
- :class:`~marapendi.models.channel.BakerChannelGasResistanceModel` —
  empirical correlations from Baker et al. (2009) for the sub-resistances.

Component
---------

.. autoclass:: marapendi.components.channel.flow_channels.FlowChannel
   :members:
   :show-inheritance:

Transport models
----------------

.. autoclass:: marapendi.models.channel.ChannelGasResistanceModel
   :members:
   :show-inheritance:

.. autoclass:: marapendi.models.channel.BakerChannelGasResistanceModel
   :members:
   :show-inheritance:
