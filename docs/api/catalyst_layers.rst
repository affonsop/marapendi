Catalyst Layers
===============

:class:`~marapendi.components.porous_layers.catalyst_layers.CatalystLayer` extends
:class:`~marapendi.components.porous_layers.porous_layers.PorousLayer` with an ionomer phase
and an electrochemical reaction model.  It provides ionomer charge resistance via
the Neyerlin/Goshtasbi distribution model and activation overpotential from
Butler-Volmer kinetics.

:class:`~marapendi.components.porous_layers.catalyst_layers.PtCCatalystLayer` adds explicit
Pt/C agglomerate geometry following Hao et al. (2015): volume fractions are derived
from the catalyst composition and the wet ionomer film thickness is updated at each
operating point.  Local O₂ transport resistance across the ionomer film (bulk + gas/Pt
and ionomer/Pt interface terms) is evaluated at each solve step.

The electrochemical reaction is parameterised by
:class:`~marapendi.models.thermo.electrochemistry.ElectrochemicalReaction`, which holds the
Butler-Volmer parameters and the Nernst correction.

Components
----------

.. autoclass:: marapendi.components.porous_layers.catalyst_layers.CatalystLayer
   :members:
   :show-inheritance:

.. autoclass:: marapendi.components.porous_layers.catalyst_layers.PtCCatalystLayer
   :members:
   :show-inheritance:

Electrochemical reaction
------------------------

.. autoclass:: marapendi.models.thermo.electrochemistry.ElectrochemicalReaction
   :members:
   :show-inheritance:
