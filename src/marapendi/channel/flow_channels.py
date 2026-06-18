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
from ..thermo.constants import GAS_CONSTANT

from ..thermo.gas import GasModel, species_indexes
from ..porous_layers.porous_layers import PorousLayer
from ..thermo.water import water_kinematic_viscosity, water_molar_volume


@dataclass
class ChannelGasResistanceModel:
    """Channel gas transport resistance using a Sherwood-number approach.

    References
    ----------
    Kim, H. et al. Int. J. Heat Mass Transf. 183, 122106 (2022).
    """

    sherwood: float = 4.0
    B_ch: float = 1.0

    def molecular_diffusion_resistance(self, channel, diffusion_coefficient):
        """Diffusion resistance via Sherwood number (s/m)."""
        return channel.hydraulic_diameter / (self.sherwood * diffusion_coefficient)

    def convection_resistance(self, channel, volume_flow_rate):
        """Convection resistance assuming average inlet/outlet concentration (s/m)."""
        return (
            self.B_ch * channel.length * channel.width
            * (1 + 1 / channel.channel_land_ratio) / 2
            * channel.n_parallel / (volume_flow_rate + 1e-12)
        )

    def total_resistance(self, channel, diffusion_coefficient, volume_flow_rate):
        """Total channel resistance: diffusion + convection (s/m)."""
        return (
            self.molecular_diffusion_resistance(channel, diffusion_coefficient)
            + self.convection_resistance(channel, volume_flow_rate)
        )

    def gas_transport_resistance(self, channel, state, species, volume_flow_rate=None):
        """Total gas transport resistance for *species* in *channel* (s/m)."""
        diffusion_coeff = GasModel.species_diffusion_coefficient(state, species)
        flow_rate = volume_flow_rate if volume_flow_rate is not None else state.inlet_gas_flow_rate
        return self.total_resistance(channel, diffusion_coeff, flow_rate)


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

