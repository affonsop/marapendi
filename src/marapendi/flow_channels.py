"""
Flow channel components: static physical properties and correlations.

:class:`FlowChannel` extends :class:`~marapendi.porous_layers.PorousLayer`
with the geometric properties of a gas flow channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .transport_models import ChannelGasResistanceModel
from .gas import GasModel
from .porous_layers import PorousLayer


@dataclass
class FlowChannel(PorousLayer):
    """Flow channel geometry of a single fuel cell side.

    Attributes
    ----------
    reactant : str
        Primary reactant species ('o2' or 'h2').
    width, height, length : float
        Channel cross-section width, height and length (m).
    channel_land_ratio : float
        Ratio of channel width to land width.
    n_parallel : int
        Number of parallel channels.
    transport_resistance_model : ChannelGasResistanceModel
        Correlation used for the channel gas transport resistance.
    """

    reactant: str = 'o2'
    width: float = 1e-3
    height: float = 1e-3
    length: float = 100e-3
    channel_land_ratio: float = 1.
    n_parallel: int = 14
    transport_resistance_model: ChannelGasResistanceModel = field(default_factory=ChannelGasResistanceModel)

    def __post_init__(self):
        self.hydraulic_diameter = 2 * self.width * self.height / (self.width + self.height)
        self.channel_flow_section = self.width * self.height
        self.total_flow_section = self.n_parallel * self.channel_flow_section

    def gas_transport_resistance(self, state, species: str = 'o2') -> float:
        """Gas transport resistance for ``species`` (s/m), given the channel's ``state``."""
        diffusion_coefficient = GasModel.species_diffusion_coefficient(state, species)
        return self.transport_resistance_model.total_resistance(
            self, diffusion_coefficient, state.inlet_gas_flow_rate,
        )
