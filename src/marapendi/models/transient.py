"""
Transient cell model for PEM/AEM electrochemical cells.

Classes
-------
TransientCellModel : ODE engine that wraps a Cell and a VoltageModel to
    assemble and integrate the full spatially-resolved transient equations
    (water content, temperature, gas concentrations, liquid saturation).
"""

import numpy as np
import cantera as ct
from dataclasses import dataclass, field

from marapendi.components.cell import Cell
from marapendi.components.cell_state import CellState
from marapendi.models.water import (
    water_saturation_concentration,
    water_density,
    water_molar_volume,
    water_kinematic_viscosity,
)
from marapendi.models.electrochemistry import enthalpy_condensation


@dataclass
class TransientCellModel:
    """
    Spatially-resolved transient ODE model for a PEM/AEM cell.

    The model assembles dxdt for the state vector
    ``x[n_layers, n_variables]`` where the variables are:

    * ``i_lmbd = 0`` : ionomer water content λ
    * ``i_T    = 1`` : temperature T [K]
    * ``i_cg   = [2,3,4,5]`` : gas concentrations [O₂, N₂, H₂, H₂O] [kmol m⁻³]
    * ``i_s    = 6`` : liquid water saturation s [–]

    Parameters
    ----------
    cell : Cell
        Fully-configured cell geometry and property arrays.
    charge : str
        Charge carrier (``'proton'`` for PEM, ``'hydroxide'`` for AEM).
    """

    cell: Cell = field(default_factory=Cell)

    def __post_init__(self):
        cell = self.cell
        self.n_layers = len(cell.layers)

        self.ionomer_layers = [cell.an.cl, cell.memb, cell.ca.cl]

        self.ionomer_domain = [l in self.ionomer_layers for l in cell.layers]
        # Catalyst layers only: membrane lambda is quasi-steady (inf capacity)
        self.cl_domain = [l in [cell.an.cl, cell.ca.cl] for l in cell.layers]

        self.porous_domain = [
            (l in cell.ca.porous_layers) or (l in cell.an.porous_layers)
            for l in cell.layers
        ]
        self.channel_domain = [l in [cell.ca.ch, cell.an.ch] for l in cell.layers]

        self.layer_thickness = np.array(
            [cell.an.ch.height]
            + [l.thickness for l in cell.an.porous_layers[::-1]]
            + [cell.memb.thickness]
            + [l.thickness for l in cell.ca.porous_layers]
            + [cell.ca.ch.height]
        )[..., np.newaxis]

        self.x_h2_mask = np.array([
            1 if layer in cell.an.porous_layers else 0
            for layer in cell.layers
        ])

        for i, layer in enumerate(cell.layers):
            layer.ix = i

        self.n_variables = (
            1   # ionomer water content
            + 1   # temperature
            + 4   # gas concentrations
            + 1   # liquid water saturation
        )
        self.i_lmbd = 0
        self.i_T    = 1
        self.i_cg   = [2, 3, 4, 5]
        self.i_s    = 6

        self.norm_factor = np.array(
            [14, 353.15, 40e-3, 40e-3, 40e-3, 40e-3, 1.]
        )[np.newaxis, :]

    # ------------------------------------------------------------------
    # State unpacking
    # ------------------------------------------------------------------

    def get_states_from_x(self, x):
        """Unpack normalised state vector into named fields."""
        lmbd = x[:, self.i_lmbd, ...]
        T    = x[:, self.i_T,    ...]
        cg_k = x[:, self.i_cg,  ...]
        s    = x[:, self.i_s,   ...]
        return lmbd, T, cg_k, s

    # ------------------------------------------------------------------
    # rates_of_change — sub-steps
    # ------------------------------------------------------------------

    def _compute_derived_quantities(self, x, i) -> CellState:
        """Unpack state vector, compute thermodynamic fields, and return a CellState."""
        cell = self.cell
        lmbd, T, cg_k, s = self.get_states_from_x(x)

        iF    = i / ct.faraday
        c_g   = np.sum(cg_k, axis=1)
        p_g   = c_g * ct.gas_constant * T
        x_g_k = cg_k / c_g[:, np.newaxis, ...]
        p_g_k = p_g[:, np.newaxis, ...] * x_g_k
        D_g_k   = cell.gas_diffusion_model.species_diffusion_coefficient(T, p_g, x_h2=self.x_h2_mask)
        c_sat = water_saturation_concentration(T)
        c_v   = cg_k[:, -1, ...]
        rh    = c_v / c_sat
        rho_l = water_density(T)
        nu_l  = water_kinematic_viscosity(T)
        M_k   = x_g_k * np.array([32., 28., 2., 18.])[np.newaxis, :, np.newaxis]
        V_w   = water_molar_volume(T)

        f_v = cell.memb_model.water_vol_fraction(
            lmbd, V_w, cell.V_ion
        )

        # Pre-sliced convenience fields
        memb_ix  = cell.memb.ix
        ca_cl_ix = cell.ca.cl.ix
        an_cl_ix = cell.an.cl.ix

        state = CellState(
            x=x,
            lmbd=lmbd, T=T, cg_k=cg_k, s=s,
            iF=iF, p_g=p_g, p_g_k=p_g_k, D_g_k=D_g_k,
            c_sat=c_sat, c_v=c_v, rh=rh,
            rho_l=rho_l, nu_l=nu_l, M_k=M_k, f_v=f_v,
            # slices
            T_memb=T[memb_ix, ...],
            T_ca_cl=T[ca_cl_ix, ...],
            T_an_cl=T[an_cl_ix, ...],
            f_v_memb=f_v[memb_ix, ...],
            f_v_ca_cl=f_v[ca_cl_ix, ...],
            lmbd_ca_cl=lmbd[ca_cl_ix, ...],
            p_h2=p_g_k[an_cl_ix, 2, ...],
            p_o2_ca_cl=p_g_k[ca_cl_ix, 0, ...],
        )

        # Local O2 partial pressure at cathode CL catalyst surface
        state.t_water_film = cell.cl_model.water_film_thickness(state.s[cell.ca.cl.ix,...], cell.ca.cl)
        state.R_o2_local = cell.cl_model.o2_ionomer_film_resistance(
            state.lmbd_ca_cl, state.T_ca_cl, cell.ca.cl, cell.memb_model, 
            cell.ca.cl.t_ion_film, state.t_water_film, coverage_ratio=0
        )

        c_o2_local = state.cg_k[cell.ca.cl.ix, 0, ...] - state.R_o2_local * state.iF / 4
        state.p_o2_local = c_o2_local * ct.gas_constant * state.T_ca_cl

        (state.V_cell, state.eta_ohm, state.eta_act,
         state.E_rev_ca, state.E_rev_an,
         state.eta_memb, _, state.eta_gdl) = cell.voltage_model.calculate_cell_voltage(
            T_an_cl=state.T_an_cl,
            T_ca_cl=state.T_ca_cl,
            T_memb=state.T_memb,
            f_v_memb=state.f_v_memb,
            f_v_ca_cl=state.f_v_ca_cl,
            s_ca_cl=0, 
            p_h2=state.p_h2,
            p_o2_local=state.p_o2_local,
            p_o2_ca_cl=state.p_o2_ca_cl,
            i=state.iF * ct.faraday,
            memb=cell.memb,
            electrical_resistance=cell.electrical_resistance,
            memb_model=cell.memb_model,
            ionomer_model=cell.memb_model,
            ca_cl_model=cell.cl_model,
            ca_cl=cell.ca.cl,
            charge=cell.charge,
        )
        
        return state

    def _compute_resistances(self, state: CellState):
        """Build layer resistance array R and harmonic-mean inter-layer eff_R."""
        cell = self.cell
        R = np.zeros_like(state.x)

        # λ is only defined in the ionomer domain; block transport everywhere else so
        # the BDF Jacobian never sees a trivially-zero diagonal for those variables.
        R[:, self.i_s, ...] = cell.darcy_transport_model.calculate_liquid_darcy_flow_resistance(
            state.s, state.nu_l, cell.thickness,
            cell.K_abs, cell.n_rel,
        )
        R[:, self.i_cg, ...] = cell.gas_diffusion_model.total_diffusion_resistance(
            state.T[:, np.newaxis, ...],
            state.s[:, np.newaxis, ...],
            state.D_g_k,
            state.M_k,
            cell.thickness[:, np.newaxis, ...],
            cell.eps_p[:, np.newaxis, ...],
            cell.tort[:, np.newaxis, ...],
            cell.d_p[:, np.newaxis, ...],
            cell.n_s[:, np.newaxis, ...],
        )
        R[:, self.i_T, ...] = cell.thickness / cell.bulk_thermal_conductivity

        # λ transport is only defined in the ionomer domain; default to inf everywhere
        # and overwrite for ionomer layers only to avoid divide-by-zero on other layers.
        
        ion = self.ionomer_domain
        D_lmbd = cell.memb_model.diffusion_coefficient(
            state.lmbd, state.f_v, state.T,
            cell.darken_num_ion,
            cell.darken_den_ion,
            cell.D_lmbd_ref_ion,
            cell.E_act_ion,
        )
        R[:, self.i_lmbd, ...] = cell.memb_model.calculate_membrane_water_resistance(
            D_lmbd, cell.thickness, cell.eps_ion,
            cell.c_ion, cell.tau_ion,
        )

        for layer in (cell.ca.ch, cell.an.ch):
            R[layer.ix, self.i_s, ...]  = 0
            R[layer.ix, self.i_cg, ...] = (
                layer.hydraulic_diameter / state.D_g_k[layer.ix, ...] / layer.sherwood
            )
            R[layer.ix, self.i_T, ...]  = 2 * layer.height / layer.bulk_thermal_conductivity

        R[cell.memb.ix, [self.i_s] + self.i_cg, ...] = np.inf
        np.nan_to_num(R, copy=False, nan=np.inf, posinf=np.inf)

        eff_R = (R[:-1, ...] + R[1:, ...]) / 2
        eff_R[-1, self.i_T, ...] += cell.thermal_resistance / 2
        eff_R[ 0, self.i_T, ...] += cell.thermal_resistance / 2
        return R, eff_R

    def _compute_fluxes(self, eff_R, state: CellState):
        """Compute inter-layer flux array J (diffusion + advection + EOD)
        and return V_cell, eta_ohm, eta_act, S_T_losses, p_o2_local."""
        cell = self.cell
        phi = state.x.copy()

        # Liquid-water driving potential: gas pressure + capillary pressure
        phi[:, self.i_s, ...] = state.p_g
        phi[self.porous_domain, self.i_s, ...] += (
            cell.darcy_transport_model.capillary_pressure_from_saturation(
                state.s,
                cell.p_b,
                cell.van_genuchten_m,
                cell.van_genuchten_n,
            )[self.porous_domain, ...]
        )

        J = np.zeros((self.n_layers + 1, self.n_variables, state.x.shape[-1]))
        J[1:-1, ...] = -(phi[1:, ...] - phi[:-1, ...]) / eff_R

       

        # Electro-osmotic drag
        J_eod = (
            cell.memb.calculate_electroosmotic_drag_coefficient(state.T, state.lmbd)
            [self.ionomer_domain, ...]
            * state.iF
        )
        J[[cell.an.cl.ix + 1, cell.memb.ix + 1], self.i_lmbd, ...] += J_eod[:1, ...]

        return J

    def _compute_water_exchange(self, state: CellState):
        """Return ionomer absorption/desorption flux J_des."""
        cell = self.cell
        k_abs = cell.memb_model.sorption_coefficient(
            state.f_v, state.T,
            cell.k_des_ref_ion,
            cell.E_act_ion,
            cell.T_ref_des_ion,
        )
        lmbd_eq = (
            (1 - state.s) * cell.memb_model.equilibrium_water_content(
                state.rh, cell.sorption_coeffs_ion,
            )
            + state.s * cell.memb_model.liquid_equilibrium_water_content(
                cell.lmbd_liq_ref_ion,
            )
        )
        return k_abs * (state.lmbd - lmbd_eq) * cell.c_ion

    def _compute_phase_change(self, state: CellState):
        """Return vapour–liquid phase-change source term S_vl.

        Positive when condensation occurs (c_v > c_sat), negative when evaporating.
        """
        c_sat  = water_saturation_concentration(state.T)
        factor = np.where(c_sat > state.c_v, state.s, 1 - state.s)
        return 1000.0 * (state.c_v - c_sat) * factor

    def _compute_sources(self, state: CellState, J_des, S_vl):
        """Populate source term array S."""
        cell = self.cell
        S = np.zeros_like(state.x)

        # Assemble per-layer thermal loss array
        memb_ix   = cell.memb.ix
        ca_cl_ix  = cell.ca.cl.ix
        an_cl_ix  = cell.an.cl.ix
        ca_gdl_ix = cell.ca.gdl.ix
        an_gdl_ix = cell.an.gdl.ix
        i = state.iF * ct.faraday

        S_T_losses = np.zeros_like(state.T)
        S_T_losses[memb_ix,   ...] = state.eta_memb  * i
        S_T_losses[ca_cl_ix,  ...] = (state.eta_act + state.E_rev_ca) * i
        S_T_losses[an_cl_ix,  ...] = state.E_rev_an  * i
        S_T_losses[ca_gdl_ix, ...] = state.eta_gdl   * i
        S_T_losses[an_gdl_ix, ...] = state.eta_gdl   * i

        # Ionomer water: absorption sink ± electrolysis production
        S[cell.an.cl.ix, self.i_lmbd, ...] += (
            -J_des[cell.an.cl.ix, ...]
        ) / cell.an.cl.thickness
        S[cell.ca.cl.ix, self.i_lmbd, ...] += (
            state.iF / 2 - J_des[cell.ca.cl.ix, ...]
        ) / cell.ca.cl.thickness

        # Liquid water: phase change (blocked in membrane)
        S[:, self.i_s, ...]            = S_vl
        S[cell.memb.ix, self.i_s, ...] = 0

        # Water vapour: condensation sink
        S[:, self.i_cg[-1], ...] = -S_vl

        # Gas species: electrolysis consumption
        S[cell.ca.cl.ix, self.i_cg[0], ...] = (-state.iF / 4) / cell.ca.cl.thickness
        S[cell.an.cl.ix, self.i_cg[2], ...] = (-state.iF / 2) / cell.an.cl.thickness

        # Water vapour: re-evaporation from absorbed water
        for side in (cell.an, cell.ca):
            S[side.cl.ix, self.i_cg[-1], ...] += J_des[side.cl.ix, ...] / side.cl.thickness
        S[cell.memb.ix, self.i_cg, ...] = 0

        # Temperature: ohmic/activation losses + phase-change enthalpy
        h_vl = enthalpy_condensation(state.T)
        S[:, self.i_T, ...] = S_T_losses / cell.thickness + S_vl * h_vl
        for side in (cell.an, cell.ca):
            S[side.cl.ix, self.i_T, ...] -= (
                J_des[side.cl.ix, ...] / side.cl.thickness
                * cell.memb_model.heat_of_adsorption(state.T[side.cl.ix, ...], cell.memb)
            )

        return S

    def _compute_capacities(self, state: CellState):
        """Populate capacity array C."""
        cell = self.cell
        C = np.ones_like(state.x)

        # Membrane lambda is quasi-steady → inf capacity; only CLs have finite dynamics
        C[:, self.i_lmbd, ...] = (
            cell.eps_ion * cell.c_ion * cell.thickness
        )
        C[:, self.i_s,  ...]   = state.rho_l * cell.eps_p
        C[:, self.i_cg, ...]   = cell.eps_p[:, np.newaxis, ...]
        C[:, self.i_T, ...]    = cell.bulk_density * cell.bulk_specific_heat_capacity
        np.nan_to_num(C, copy=False, nan=np.inf)
        print(C[:, self.i_lmbd, ...])
        return C

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def rates_of_change(self, x, i=0.):
        """Return dxdt for all state variables across all layers.

        Parameters
        ----------
        x : ndarray, shape (n_layers * n_variables, m)
        i : float
        """
        cell = self.cell
        x = (
            x.reshape(self.n_layers, self.n_variables, x.shape[-1])
            * self.norm_factor[..., np.newaxis]
        )

        # 1. Derived thermodynamic quantities → CellState
        state = self._compute_derived_quantities(x, i)

        # 2. Transport resistances
        R, eff_R = self._compute_resistances(state)

        # 3. Inter-layer fluxes (including EOD)
        J = (
            self._compute_fluxes(eff_R, state)
        )

        # 4. Water exchange between phases
        J_des = self._compute_water_exchange(state)
        S_vl  = self._compute_phase_change(state)

        # 5. Source terms
        S = self._compute_sources(state, J_des, S_vl)

        # 6. Capacities
        C = self._compute_capacities(state)

        # 7. Assemble dxdt; enforce channel boundary conditions
        dxdt = ((J[:-1, ...] - J[1:, ...]) / cell.thickness[:, np.newaxis] + S) / C
        for ch in (cell.ca.ch, cell.an.ch):
            dxdt[ch.ix, self.i_T,  :] = 0
            dxdt[ch.ix, self.i_cg, :] = 0
            dxdt[ch.ix, self.i_s,  :] = 0

        return dxdt.reshape(self.n_layers * self.n_variables, state.x.shape[-1])
