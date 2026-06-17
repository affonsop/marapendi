"""
Module providing channel transport models.
"""
from dataclasses import dataclass

@dataclass 
class BakerChannelGasResistanceModel:
    """
    Empirical model for gas transport resistance in flow channels. 
    Based on the work of Baker et al. (2009). 

    Attributes
    ----------
    A_ch : float
        Coefficient for diffusion resistance.
    B_ch : float
        Coefficient for convection resistance.

    References
    ----------
    Baker et al. J. Electrochem. Soc. 156, B991 (2009).
    """
    A_ch: float = 1.0
    B_ch: float = 1.0

    def molecular_diffusion_resistance(self, channel, diffusion_coefficient):
        """
        Compute diffusion resistance in the channel.

        Parameters
        ----------
        channel : FlowChannel
            Flow channel object with geometry.
        diffusion_coefficient : float
            Molecular diffusion coefficient [m2/s].

        Returns
        -------
        float
            Channel diffusion resistance [s/m].
        """
        return self.A_ch * channel.half_width / diffusion_coefficient
    
    def convection_resistance(self, channel, volume_flow_rate):
        """
        Compute convection resistance in the channel.

        Parameters
        ----------
        channel : FlowChannel
            Flow channel object with geometry.
        volume_flow_rate : float
            Volumetric flow rate [m3/s].

        Returns
        -------
        float
            Channel convection resistance [s/m].
        """
        return self.B_ch * channel.length / channel.half_width * channel.total_flow_section / (volume_flow_rate + 1e-12)

    def total_resistance(self, channel, diffusion_coefficient, volume_flow_rate):
        """
        Compute total gas transport resistance (diffusion + convection).

        Parameters
        ----------
        channel : FlowChannel
            Flow channel object.
        diffusion_coefficient : float
            Molecular diffusion coefficient [m2/s].
        volume_flow_rate : float
            Volumetric flow rate [m3/s].

        Returns
        -------
        float
            Total channel resistance [s/m].
        """
        return (self.molecular_diffusion_resistance(channel, diffusion_coefficient) +
                self.convection_resistance(channel, volume_flow_rate))

@dataclass
class ChannelGasResistanceModel:
    """
    Channel gas transport resistance model using Sherwood number approach.
    Similar equations are used by Kim et al. (2022). 

    Attributes
    ----------
    sherwood : float
        Sherwood number for mass transfer.
    B_ch : float
        Coefficient for convection resistance.
    
    References
    ----------
    Kim, H. et al. Int. J. Heat Mass Transf. 183, 122106 (2022).
    """
    sherwood: float = 4.0
    B_ch: float = 1.0

    def molecular_diffusion_resistance(self, channel, diffusion_coefficient):
        """
        Compute molecular diffusion resistance using Sherwood number.

        Parameters
        ----------
        channel : FlowChannel
            Channel geometry.
        diffusion_coefficient : float
            Diffusion coefficient [m2/s].

        Returns
        -------
        float
            Channel diffusion resistance [s/m].
        """
        return 1/self.sherwood * channel.hydraulic_diameter / diffusion_coefficient
    
    def convection_resistance(self, channel, volume_flow_rate):
        """
        Compute convection resistance with channel geometry.
        Assumes that the channel gas concentration is equal to the average between 
        inlet and outlet values.  

        Parameters
        ----------
        channel : FlowChannel
            Channel geometry.
        volume_flow_rate : float
            Volumetric flow rate [m3/s].

        Returns
        -------
        float
            Channel convection resistance [s/m].
        """
        return self.B_ch * channel.length * channel.width * (1 + 1/channel.channel_land_ratio) / 2 * channel.n_parallel / (volume_flow_rate + 1e-12)

    def total_resistance(self, channel, diffusion_coefficient, volume_flow_rate):
        """
        Compute total resistance combining diffusion and convection.

        Parameters
        ----------
        channel : FlowChannel
            Channel geometry.
        diffusion_coefficient : float
            Diffusion coefficient [m2/s].
        volume_flow_rate : float
            Volumetric flow rate [m3/s].

        Returns
        -------
        float
            Channel total resistance [s/m].
        """
        return (self.molecular_diffusion_resistance(channel, diffusion_coefficient) +
                self.convection_resistance(channel, volume_flow_rate))
    
