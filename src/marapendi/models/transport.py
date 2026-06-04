"""
Module providing gas and liquid transport models.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
import cantera as ct
from marapendi.models.water import water_surface_tension, water_saturation_concentration, water_molecular_weight

if TYPE_CHECKING:
    from marapendi.models.transient import TransientCellModel
    from marapendi.components.cell_state import CellState



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

    
@dataclass
class PorousGasResistanceModel:
    """
    Porous media gas transport model with water saturation and Knudsen corrections.

    Attributes
    ----------
    n_s : float
        Exponent for empirical water saturation correction.
    """
    n_s: float = 3.0

    def species_diffusion_coefficient(self, temperature, pressure, x_h2=0):
        """
        Calculate the binary diffusion coefficient for a given species in the gas phase.

        Uses empirical correlations based on reference values adjusted for
        temperature and pressure. Data from Vetter and Schumacher (2019).

    
        Returns
        -------
        float
            The adjusted diffusion coefficient [m^2/s].

        Reference
        ----------
        Vetter, R. & Schumacher, J. O. Comput. Phys. Commun. 234, 223–234 (2019).
        """
        ("O2", "N2", "H2", "H2O")
        reference_diffusion_coeff =np.where(x_h2[...,np.newaxis] <= 0, np.array([0.28e-4, 0.28e-4, 0.28e-4, 0.36e-4])[np.newaxis, ...] ,
                                    np.array([1.24e-4, 1.24e-4, 1.24e-4, 1.24e-4])[np.newaxis, ...])


        # Apply temperature and pressure correction
        # Fick's law adjustment: D ~ T^1.5 / P
        diffusion_coeff = reference_diffusion_coeff[...,np.newaxis] * (temperature ** 1.5 / np.maximum(pressure, 1.) * 15.0682)[:,np.newaxis,...]
        return diffusion_coeff

    def water_saturation_correction(self, s, n_s):
        """
        Compute correction factor for effective diffusivity.

        Parameters
        ----------
        s : float
            Water saturation [-].
        n_s : float
            Water saturation exponent [-].
        Returns
        -------
        float
            Correction factor [-].
        """
        return np.clip(1 - s, 1e-6, 1) ** n_s

    def molecular_diffusion_effective_length(self, thickness, porosity, tortuosity, s, n_s):
        return thickness * tortuosity / porosity / self.water_saturation_correction(s, n_s)

    def molecular_diffusion_resistance(self, D_g_k, thickness, porosity, tortuosity, s, n_s):
        return self.molecular_diffusion_effective_length(thickness, porosity, tortuosity, s, n_s) / D_g_k

    def knudsen_diffusivity(self, T, d_p, M_k):
        return d_p / 3 * np.sqrt(8 * ct.gas_constant * T / np.maximum(M_k, 1e-30) / np.pi)

    def total_diffusion_resistance(self, T, water_saturation, D_g_k, species_molecular_weights, thickness, porosity, tortuosity, pore_diameter, water_saturation_exponent):
        """
        Compute total diffusion resistance (molecular + Knudsen).

        Returns
        -------
        float
            Total resistance [s/m].
        """
        return self.molecular_diffusion_effective_length(thickness, porosity, tortuosity, water_saturation, water_saturation_exponent) * (
            1/D_g_k + 1/self.knudsen_diffusivity(T, pore_diameter, species_molecular_weights))

    def update_transport_matrices(self, state: CellState, cell, tm: TransientCellModel, gas_model) -> None:
        """
        Fill state.R[:,i_cg], state.C[:,i_cg], state.S[:,i_cg].
        Reads state.S_lv (must be set by DarcyTransportModel first).

        Parameters
        ----------
        gas_model :
            Gas mixture model providing ``molecular_weights`` and other
            species properties.
        """
        import numpy as np
        i_cg = tm.i_cg

        # Diffusion resistance (porous layers)
        state.R[:, i_cg, ...] = self.total_diffusion_resistance(
            state.T[:, np.newaxis, ...],
            state.s[:, np.newaxis, ...],
            state.D_g_k,
            gas_model.molecular_weights[np.newaxis, :, np.newaxis],
            cell.thickness[:, np.newaxis, ...],
            cell.eps_p[:, np.newaxis, ...],
            cell.tort[:, np.newaxis, ...],
            cell.d_p[:, np.newaxis, ...],
            cell.n_s[:, np.newaxis, ...],
        )
        # Channel: Sherwood-number mass-transfer resistance
        for ch in (cell.ca.ch, cell.an.ch):
            state.R[ch.ix, i_cg, ...] = (
                ch.hydraulic_diameter / state.D_g_k[ch.ix, ...] / ch.sherwood
            )
        # Membrane: impermeable to gas
        state.R[cell.memb.ix, i_cg, ...] = np.inf

        # Capacity
        state.C[:, i_cg, ...] = cell.eps_p[:, np.newaxis, ...]

        # Sources: reactant consumption
        state.S[cell.ca.cl.ix, i_cg[0],  ...] += (-state.iF / 4) / cell.ca.cl.thickness  # O2
        state.S[cell.an.cl.ix, i_cg[2],  ...] += (-state.iF / 2) / cell.an.cl.thickness  # H2
        state.S[:, i_cg[-1], ...]              += -state.S_lv  # vapour condensation
        state.S[cell.memb.ix, i_cg, ...]        = 0.           # no gas in membrane

@dataclass
class DarcyTransportModel: 
    """
    Model for calculating non-wetting phase transport in porous layers using a Darcy-based approach.

    Attributes
    ----------
    J_function_exponent : float
        Exponent in the non-wetting phase capillary pressure-saturation relation.
    """
 

    def calculate_darcy_flow_resistance(self, s, nu, thickness, K_abs, n_rel):
        # on a mass basis
        return ((thickness * nu) /
                (K_abs * np.maximum(1e-2, s) ** n_rel))

    def saturation_from_capillary_pressure(self, layer, p_c, p_b, m, n, phase='non-wetting'):
        """
        Van Genuchten model
        """
        if phase == 'non-wetting': 
            return 1 - (1 + (p_c/p_b)**n)**(-m)
        else: 
            return (1 + (p_c/p_b)**n)**(-m) 

    def capillary_pressure_from_saturation(self, s, p_b, m, n, phase='non-wetting'):
        """
        Van Genuchten model
        """
        return  p_b * (
                np.clip(1 - s if phase == 'non-wetting' else s, 1e-6, 1)
                **(-1/m) - 1
            )**(1/n)

    def update_transport_matrices(self, state: CellState, cell, tm: TransientCellModel) -> None:
        """
        Fill state.phi[:,i_s], state.R[:,i_s], state.C[:,i_s], state.S[:,i_s].
        Computes and stores state.S_lv (vapour-liquid phase-change rate).
        """
        import numpy as np
        i_s = tm.i_s

        # Driving potential: gas pressure + capillary pressure (overrides default phi = x)
        porous_ix = cell.ca.ix + cell.an.ix
        state.phi[:, i_s, ...] = state.p_g
        state.phi[porous_ix, i_s, ...] += (
            self.capillary_pressure_from_saturation(
                state.s, cell.p_b, cell.van_genuchten_m, cell.van_genuchten_n,
            )[porous_ix, ...]
        )

        # Resistance
        state.R[:, i_s, ...] = self.calculate_darcy_flow_resistance(
            state.s, state.nu_l, cell.thickness, cell.K_abs, cell.n_rel,
        )
        state.R[cell.memb.ix, i_s, ...] = np.inf
        for ch in (cell.ca.ch, cell.an.ch):
            state.R[ch.ix, i_s, ...] = 0.

        # Capacity
        state.C[:, i_s, ...] = state.rho_l * cell.eps_p

        # Phase-change source (stored on state for ThermalModel and GasModel)
        c_sat = water_saturation_concentration(state.T)
        factor = np.where(c_sat > state.c_v, state.s, 1 - state.s)
        state.S_lv = 1000.0 * (state.c_v - c_sat) * factor

        state.S[:, i_s, ...]           += state.S_lv * water_molecular_weight
        state.S[cell.memb.ix, i_s, ...]  = 0.
