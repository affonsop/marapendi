from dataclasses import dataclass
import numpy as np
import cantera as ct 

from marapendi.models.transient import TransientCellModel
from marapendi.components.cell_state import CellState
from marapendi.components.operating_conditions import OperatingConditions

@dataclass 
class GasFlowFieldModel: 
    def gas_to_liquid_slip_ratio(self, mu_g, mu_l, s_l): 
        return (s_l / (1-s_l)) ** 3 * mu_g / mu_l

    def calculate_inlet_gas_pressure(self, state, gfc, conditions): 
        # Does not account for two-phase flow. 
        p_ch = state.p_g[gfc.ix, ...]
        RT_in = ct.gas_constant * conditions.temperature 

        # We first calculate the pressure drop assuming the p_in = p_ch
        vol_flow_rate_ch = conditions.inlet_gas_molar_flow_rate / (p_ch / RT_in) 
        dpdx_ch = (gfc.fRe * state.mu_g) / (gfc.hydraulic_diameter ** 2 / 2) * gfc.superficial_velocity(vol_flow_rate_ch) 
        delta_p_ch = dpdx_ch * gfc.length / 2

        # Then we correct based on the equation p_in = p_ch + delta_p_ch p_ch / p_in since delta_p is inversely proportional to
        # the pressure 
        p_in = p_ch * (1 + np.sqrt(1 + 4 * delta_p_ch / p_ch)) / 2

        conditions.inlet_pressure = p_in 
        return p_in 
    
    def calculate_outlet_flows(self, state, gfc, conditions): 
        # Based on pressure drop and friction factor to determine the outlet velocities. 
        pressure_gradient =  2 * (state.p_g[gfc.ix, ...] - conditions.backpressure) / gfc.length 
        u_g_out = pressure_gradient * (gfc.hydraulic_diameter ** 2 / 2) / (gfc.fRe * state.mu_g[gfc.ix, ...]) 
        u_l_out = self.gas_to_liquid_slip_ratio(state.mu_g[gfc.ix, ...], state.mu_l[gfc.ix, ...], state.s[gfc.ix, ...]) * u_g_out
        n_dot_k_out = state.cg_k[gfc.ix, ...] * u_g_out[np.newaxis,...] * gfc.total_flow_section
        m_dot_l_out = u_l_out * state.s[gfc.ix, ...] * state.rho_l[gfc.ix, ...] * gfc.total_flow_section
        return n_dot_k_out, m_dot_l_out 
    
    def add_channel_inlet_outlet_flows(self, conditions, state, gfc, tm):
        n_dot_k_out, m_dot_l_out = self.calculate_outlet_flows(state, gfc, conditions)
        inlet = conditions.inlet_gas_molar_flow_rates
        if inlet.ndim == 1:
            inlet = inlet[:, np.newaxis]  # (n_species,) → (n_species, 1) for broadcast vs (n_species, m)
        state.S[gfc.ix, tm.i_cg, ...] += ((inlet - n_dot_k_out) / gfc.total_volume)
        state.S[gfc.ix, tm.i_s, ...] += ((conditions.inlet_liquid_mass_flow_rate - m_dot_l_out) / gfc.total_volume)

    def update_channel_resistances(self, state, cell, tm): 
        # The approach described here is a bit different from the one in Yang et al. (2019), which would correspond to modifying 
        # the effective resistances. It corresponds to a case with zero in-plane resistance, whereas in Yang et al. (2019), their 
        # approach correspond to infinite in-plane resistnaces. 

        for side in cell.sides: 
            state.R[side.ch.ix, tm.i_cg + [tm.i_s], ...] *= side.channel_to_cell_area_ratio # Mass flows only through channels 
            state.R[side.ch.ix, tm.i_T, ...] *= (1-side.channel_to_cell_area_ratio) # Heat flows only through ribs 

    def update_transport_matrices(
        self,
        state: CellState,
        cell,
        tm: TransientCellModel,
        cathode_conditions: OperatingConditions,
        anode_conditions: OperatingConditions,
        *,
        override_phi: bool = True,
    ) -> None:
        """Set phi, R, C for both channel layers from current conditions.

        Must be called AFTER state.phi is initialised (= x.copy()) so that
        phi entries for non-channel layers are already correct.

        Parameters
        ----------
        state : CellState
        cell : Cell
        tm : TransientCellModel  — provides variable indices and norm_factor
        ca_cond, an_cond : OperatingConditions — frozen snapshots at current t
        override_phi : bool
            If True (default), overwrite phi and state.T at channel rows with
            values derived from *ca_cond* / *an_cond*.  Set to False in
            post-processing contexts where phi = x.copy() already represents
            the converged channel state and should not be replaced.
        """
        
        i_T  = tm.i_T
        i_cg = tm.i_cg
        i_s  = tm.i_s
        i_lmbd = tm.i_lmbd
        nf   = tm.norm_factor   # shape (1, n_variables)

        for ch, cond in ((cell.ca.ch, cathode_conditions), (cell.an.ch, anode_conditions)):
            ix = ch.ix

            if override_phi:
                # --- phi: channel boundary values (normalized) ---
                state.phi[ix, i_T,  ...] = cond.temperature

                # --- Diagnostic: keep state.T consistent with phi ---
                state.T[ix, ...] = cond.temperature

            # --- R: channel transport resistances ---
            state.R[ix, i_T,  ...] = 2 * ch.height / ch.bulk_thermal_conductivity
            state.R[ix, i_cg, ...] = ch.hydraulic_diameter / state.D_g_k[ix, ...] / ch.sherwood
            state.R[ix, i_s,  ...] = 0.

            # --- C: infinite capacity (reservoir — dxdt = 0 automatically) ---
            state.C[ix, i_T, ...]  = np.inf
            state.C[ix, i_lmbd, ...]  = np.inf
    
            # state.C[ix, i_s, ...]  = np.inf
        

        for side, conditions in ((cell.ca, cathode_conditions), (cell.an, anode_conditions)): 
            self.add_channel_inlet_outlet_flows(
                conditions,
                state, side.ch, tm)
