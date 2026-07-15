Porous Layers
=============

:class:`~marapendi.components.porous_layers.porous_layers.PorousLayer` is the base class for
all porous components in the electrode stack (GDL, MPL).  It holds static geometry
(thickness, porosity, absolute permeability, contact angle) and delegates transport
calculations to two pluggable model objects:

- :class:`~marapendi.models.diffusion.PorousGasDiffusionModel` — effective
  gas diffusivity with saturation and Knudsen corrections.
- :class:`~marapendi.models.darcy.DarcyTransportModel` — capillary
  pressure–saturation relationship via the J-function power law.

Components
----------

.. autoclass:: marapendi.components.porous_layers.porous_layers.PorousLayer
   :members:
   :show-inheritance:

.. autoclass:: marapendi.components.porous_layers.porous_layers.GasDiffusionLayer
   :members:
   :show-inheritance:

.. autoclass:: marapendi.components.porous_layers.porous_layers.MicroPorousLayer
   :members:
   :show-inheritance:

Transport models
----------------

.. autoclass:: marapendi.models.darcy.DarcyTransportModel
   :members:
   :show-inheritance:

.. autoclass:: marapendi.models.diffusion.PorousGasDiffusionModel
   :members:
   :show-inheritance:
