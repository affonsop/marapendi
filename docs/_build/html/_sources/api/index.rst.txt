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

   marapendi.cell.fuelcell.FuelCell
   marapendi.cell.fuelcell.FuelCellSide
   marapendi.cell.state.CellState
   marapendi.cell.state.CellSideState
   marapendi.cell.state.CatalystLayerState
   marapendi.cell.state.MembraneState

**Porous layers**

.. autosummary::

   marapendi.porous_layers.porous_layers.PorousLayer
   marapendi.porous_layers.porous_layers.GasDiffusionLayer
   marapendi.porous_layers.porous_layers.MicroPorousLayer
   marapendi.porous_layers.darcy.DarcyTransportModel
   marapendi.porous_layers.diffusion.PorousGasDiffusionModel

**Catalyst layers**

.. autosummary::

   marapendi.porous_layers.catalyst_layers.CatalystLayer
   marapendi.porous_layers.catalyst_layers.PtCCatalystLayer
   marapendi.thermo.electrochemistry.ElectrochemicalReaction

**Flow channels**

.. autosummary::

   marapendi.channel.flow_channels.FlowChannel
   marapendi.channel.gas_transport_resistance.ChannelGasResistanceModel
   marapendi.channel.gas_transport_resistance.BakerChannelGasResistanceModel

**Membrane and ionomer**

.. autosummary::

   marapendi.membrane.ionomer_base.Ionomer
   marapendi.membrane.pem.PFSAIonomer
   marapendi.membrane.membrane_base.Membrane
   marapendi.membrane.pem.PFSA
   marapendi.water_balance.water_balance.WaterBalanceModel
   marapendi.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise
   marapendi.water_balance.membrane.MembraneWaterBalanceModel

**Models**

.. autosummary::

   marapendi.cell.explicit_steady_state.ExplicitSteadyStateModel
   marapendi.cell.implicit_steady_state.ImplicitSteadyStateModel
   marapendi.cell.transient.TransientModel

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
