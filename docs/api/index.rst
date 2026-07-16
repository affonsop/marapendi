API Reference
=============

.. toctree::
   :maxdepth: 1

   cell
   porous_layers
   catalyst_layers
   channels
   membrane
   models
   conditions
   estimation
   tools

Quick class summary
-------------------

**Cell**

.. autosummary::

   marapendi.components.cell.fuelcell.FuelCell
   marapendi.components.cell.fuelcell.FuelCellSide
   marapendi.simulation.state.CellState
   marapendi.simulation.state.CellSideState
   marapendi.simulation.state.CatalystLayerState
   marapendi.simulation.state.MembraneState

**Porous layers**

.. autosummary::

   marapendi.components.porous_layers.porous_layers.PorousLayer
   marapendi.components.porous_layers.porous_layers.GasDiffusionLayer
   marapendi.components.porous_layers.porous_layers.MicroPorousLayer
   marapendi.models.darcy.DarcyTransportModel
   marapendi.models.diffusion.PorousGasDiffusionModel

**Catalyst layers**

.. autosummary::

   marapendi.components.porous_layers.catalyst_layers.CatalystLayer
   marapendi.components.porous_layers.catalyst_layers.PtCCatalystLayer
   marapendi.models.thermo.electrochemistry.ElectrochemicalReaction

**Flow channels**

.. autosummary::

   marapendi.components.channel.flow_channels.FlowChannel
   marapendi.models.channel.ChannelGasResistanceModel
   marapendi.models.channel.BakerChannelGasResistanceModel

**Membrane and ionomer**

.. autosummary::

   marapendi.components.membrane.ionomer_base.Ionomer
   marapendi.components.membrane.pem.PFSAIonomer
   marapendi.components.membrane.membrane_base.Membrane
   marapendi.components.membrane.pem.PFSA
   marapendi.models.water_balance.water_balance.WaterBalanceModel
   marapendi.models.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise
   marapendi.models.water_balance.membrane.MembraneWaterBalanceModel

**Models**

.. autosummary::

   marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel
   marapendi.models.base.implicit_steady_state.ImplicitSteadyStateModel
   marapendi.models.base.transient.TransientModel

**Operating conditions**

.. autosummary::

   marapendi.simulation.conditions.CellConditions
   marapendi.simulation.conditions.SideConditions

**Parameter estimation**

.. autosummary::

   marapendi.estimation.base_calibration.BaseModelCalibration
   marapendi.estimation.polarization_curve_calibration.SteadyStatePolarizationCurveCalibration
   marapendi.estimation.parameters.Parameter
   marapendi.estimation.parameters.UnknownParameter
