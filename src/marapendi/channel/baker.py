"""Baker et al. empirical channel gas-transport resistance model."""
from dataclasses import dataclass

from .flow_channels import ChannelGasResistanceModel


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
