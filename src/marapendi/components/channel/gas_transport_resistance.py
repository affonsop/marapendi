"""Channel gas-transport resistance models.

:class:`ChannelGasResistanceModel` uses a Sherwood-number approach.
:class:`BakerChannelGasResistanceModel` replaces the sub-resistances with the
empirical correlations from Baker et al. (2009).
"""
from __future__ import annotations

from dataclasses import dataclass

from ...models.thermo.gas import GasModel


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
class BakerChannelGasResistanceModel(ChannelGasResistanceModel):
    """Empirical channel gas transport resistance (Baker et al. 2009).

    Overrides the diffusion and convection sub-resistances with the
    empirical correlations from Baker et al.; :meth:`total_resistance`
    is inherited from :class:`ChannelGasResistanceModel`.

    Attributes
    ----------
    A_ch : float
        Coefficient for the diffusion resistance term.
    B_ch : float
        Coefficient for the convection resistance term (inherited).

    References
    ----------
    Baker et al. J. Electrochem. Soc. 156, B991 (2009).
    """

    A_ch: float = 1.0

    def molecular_diffusion_resistance(self, channel, diffusion_coefficient):
        """Diffusion resistance using Baker's half-width correlation (s/m)."""
        return self.A_ch * channel.half_width / diffusion_coefficient

    def convection_resistance(self, channel, volume_flow_rate):
        """Convection resistance using Baker's length/half-width formula (s/m)."""
        return (
            self.B_ch * channel.length / channel.half_width
            * channel.total_flow_section / (volume_flow_rate + 1e-12)
        )
