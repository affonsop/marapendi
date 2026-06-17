"""
Flow channel components: static geometry and runtime state.

:class:`FlowChannel` holds the static geometry of a fuel-cell flow channel
(dimensions, number of parallel channels) together with the runtime state
that must live on the component for the current explicit-state architecture
(inlet flow rates, stoichiometry, gas composition).  Pure computations that
derive quantities from that state are collected in :class:`FlowChannelModel`
so they can be reused independently of the component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from ...models.constants import GAS_CONSTANT

from ...models.gas import GasModel, species_indexes
from ..porous.porous_layers import PorousLayer
from ...models.channel.channel import ChannelGasResistanceModel
from ...models.water import water_kinematic_viscosity, water_molar_volume

@dataclass
class FlowChannel(PorousLayer):
    """Flow channel geometry and state of a single fuel-cell side.

    Inherits porous-layer geometry from :class:`~marapendi.porous_layers.PorousLayer`
    so that it can be used in the same gas-transport pipeline.

    Attributes
    ----------
    reactant : str
        Primary reactant species ('o2' or 'h2').
    inlet_stoichiometry : float
        Inlet stoichiometric ratio (actual / required flow).
    inlet_gas_flow_rate : float
        Volumetric gas flow rate at the channel inlet (m³/s).
    inlet_liquid_flow_rate : float
        Liquid flow rate at the channel inlet (m³/s).
    inlet_liquid_saturation : float
        Liquid saturation fraction at the inlet.
    width, height, length : float
        Channel cross-section width, height, and length (m).
    channel_land_ratio : float
        Ratio of channel width to land width.
    n_parallel : int
        Number of parallel channels.
    transport_resistance_model : ChannelGasResistanceModel
        Correlation for the channel gas transport resistance.
    """

    reactant: str = 'o2'
    inlet_stoichiometry: float = 0
    inlet_gas_flow_rate: float = 1e-12
    inlet_liquid_flow_rate: float = 0
    inlet_liquid_saturation: float = 0
    width: float = 1e-3
    height: float = 1e-3
    length: float = 100e-3
    channel_land_ratio: float = 1.
    n_parallel: int = 14
    transport_resistance_model: ChannelGasResistanceModel = field(default_factory=ChannelGasResistanceModel)

    def __post_init__(self):
        self.RT = GAS_CONSTANT * self.temperature
        self.hydraulic_diameter = 2 * self.width * self.height / (self.width + self.height)
        self.channel_flow_section = self.width * self.height
        self.half_width = 0.5 * self.width
        self.total_flow_section = self.n_parallel * self.channel_flow_section
        PorousLayer.__post_init__(self)

    def gas_transport_resistance(self, state, species=None, volume_flow_rate=None):
        """Gas transport resistance for ``species`` in the channel (s/m).

        Parameters
        ----------
        state : object
            State object with ``temperature``, ``pressure``, and ``gas.X``.
            Pass the channel itself when no separate state is available.
        species : str, optional
            Species identifier ('o2', 'h2', 'h2o').
        volume_flow_rate : float, optional
            Volumetric flow rate (m³/s); overrides the channel's inlet flow rate.
        """
        diffusion_coeff = GasModel.species_diffusion_coefficient(state, species)
        return self.transport_resistance_model.total_resistance(
            self, diffusion_coeff, volume_flow_rate)