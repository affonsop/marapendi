"""
Transient cell model for PEM/AEM electrochemical cells.

Classes
-------
TransientCellModel : ODE engine that integrates the full spatially-resolved
    transient equations (water content, temperature, gas concentrations,
    liquid saturation).  Physics models are resolved from the parent
    ``CellBaseModel`` that injects itself via ``base_model``.
    The applied current density is held in the ``current_density`` field and
    exposed to ``BaseModel`` through ``get_inputs``.
"""

import numpy as np
import cantera as ct
from dataclasses import dataclass, field
from typing import Union, Callable

from marapendi.components.cell import Cell
from marapendi.components.cell_state import CellState

from marapendi.models.water import (
    water_saturation_concentration,
    water_density,
    water_molar_volume,
    water_kinematic_viscosity,
    water_molecular_weight,
)
from marapendi.models.electrochemistry import enthalpy_condensation, std_formation_entropy_h2ol


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

    Physics models are resolved from the parent ``CellBaseModel`` via
    ``base_model``, injected automatically during ``CellBaseModel.__post_init__``.
    Do not set ``base_model`` manually.

    Parameters
    ----------
    cell : Cell
        Cell geometry and material property arrays (no model objects).
    current_density : float or callable
        Applied current density [A m⁻²].  May be a scalar (constant) or a
        callable ``f(t) -> float`` for time-varying protocols.  Exposed to
        ``CellBaseModel`` via ``get_inputs`` so no manual ``input_fns`` dict
        is needed.  Default: 0.
    """

    cell: Cell = field(default_factory=Cell)
    current_density: Union[float, Callable[[float], float]] = 0.
    # Injected by CellBaseModel.__post_init__; never set by the caller.
    base_model: object = field(default=None, repr=False, compare=False)

    def get_inputs(self, t: float) -> dict:
        """Return ``{'i': current_density}`` evaluated at time *t*."""
        i = self.current_density(t) if callable(self.current_density) else float(self.current_density)
        return {'i': i}

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

        for side in (cell.ca, cell.an): 
            side.ix = [layer.ix for layer in side.layers]
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

    @property
    def n_states(self) -> int:
        """Total length of the normalised flat state vector (n_layers × n_variables)."""
        return self.n_layers * self.n_variables

    # ------------------------------------------------------------------
    # Initial-state helper
    # ------------------------------------------------------------------

    def initial_state(
        self,
        cell_temperature: float = 353.15,
        cell_pressure: float = 1e5,
        ca_rh: float = 0.7,
        an_rh: float = 0.7, 
        ca_dry_o2: float = 0.21,
        ca_dry_h2: float = 0, 
        an_dry_o2: float = 0.,
        an_dry_h2: float = 1.0,
        s_ca: float = 0, 
        s_an: float = 0
    ) -> np.ndarray:
        
        """Build a normalised flat initial-state vector from operating conditions.

        Ionomer water content is initialised at the equilibrium value for the
        average of the two inlet relative humidities.  Liquid saturation starts
        at zero.  Gas compositions are assigned per-side via the ideal-gas law.

        The returned vector is normalised by ``self.norm_factor`` and flattened,
        making it suitable for direct use as ``y0`` in
        ``scipy.integrate.solve_ivp``.

        Parameters
        ----------
        cell_temperature : float
            Operating temperature [K] (default 353.15 K = 80 °C).
        cell_pressure : float
            Total gas pressure [Pa] (default 1e5 Pa = 1 bara).
        ca_rh : float
            Cathode inlet relative humidity, 0–1 (default 0.7).
        an_rh : float
            Anode inlet relative humidity, 0–1 (default 0.7).
        ca_dry_o2 : float
            O₂ mole fraction in the dry cathode gas (default 0.21 for air).
        ca_dry_h2 : float
            H₂ mole fraction in the dry cathode gas (default 0, none in air).
        an_dry_o2 : float
            O₂ mole fraction in the dry anode gas (default 0).
        an_dry_h2 : float
            H₂ mole fraction in the dry anode gas (default 1.0 for pure H₂).
        s_ca : float
            Cathode saturation (default 0).
        s_an : float
            Anode saturation (default 0).
        Returns
        -------
        np.ndarray
            Normalised flat state vector of shape ``(n_layers * n_variables,)``.
        """
        from marapendi.models.water import water_saturation_pressure

        cell  = self.cell
        RT    = ct.gas_constant * cell_temperature
        p_v = ca_rh * water_saturation_pressure(cell_temperature)
        p_dry  = cell_pressure - p_v
        c_v = p_v / RT

        # Species order in state vector: [O₂, N₂, H₂, H₂O]
        c_ca = np.array([
            ca_dry_o2       * p_dry / RT,   # O₂
            (1 - ca_dry_o2 - ca_dry_h2) * p_dry / RT,   # N₂
            ca_dry_h2,                             # H₂  (trace)
            c_v,
        ])
        c_an = np.array([
            an_dry_o2,                             # O₂  (trace)
            (1 - an_dry_h2-an_dry_o2) * p_dry / RT,   # N₂  (if any)
            an_dry_h2       * p_dry / RT,   # H₂
            c_v,
        ])

        x0 = np.zeros((self.n_layers, self.n_variables))
        rh_mean = np.full(self.n_layers, (ca_rh + an_rh) / 2)
        lmbd_eq = self.base_model.memb_model.equilibrium_water_content(
            rh_mean, cell.sorption_coeffs_ion,
        )[:, 0]
        # Non-ionomer layers (GDL, channel) return NaN — zero is fine since
        # those lambda states are frozen (infinite capacity) during integration.
        x0[:, self.i_lmbd] = np.nan_to_num(lmbd_eq, nan=0.)
        x0[:, self.i_T]    = cell_temperature
        x0[cell.ca.ix, self.i_s]    = s_ca
        x0[cell.an.ix, self.i_s]    = s_an

        memb_ix = cell.memb.ix
        for layer in cell.layers:
            if layer is cell.memb:
                continue
            x0[layer.ix, self.i_cg] = c_ca if layer.ix > memb_ix else c_an

        return (x0 / self.norm_factor).flatten()

    # ------------------------------------------------------------------
    # State unpacking
    # ------------------------------------------------------------------

    def get_states_from_x(self, x):
        """Unpack normalised state vector into named fields.

        Physical bounds are enforced here so that model functions
        (Cantera calls, log/sqrt) never receive unphysical inputs,
        even when the BDF solver evaluates Jacobian trial steps outside
        the feasible region.
        """
        lmbd = x[:, self.i_lmbd, ...]
        T    = np.clip(x[:, self.i_T, ...], 275., 600.)
        cg_k = np.maximum(x[:, self.i_cg, ...], 0.)
        s    = np.clip(x[:, self.i_s, ...], 0., 1.)
        return lmbd, T, cg_k, s

    # ------------------------------------------------------------------
    # rates_of_change — sub-steps
    # ------------------------------------------------------------------

    def _compute_derived_quantities(self, x, i) -> CellState:
        """Unpack state vector and compute thermodynamic fields."""
        cell = self.cell
        m    = self.base_model
        lmbd, T, cg_k, s = self.get_states_from_x(x)

        RT = ct.gas_constant * T
        iF    = i / ct.faraday
        c_g   = np.sum(cg_k, axis=1)
        p_g   = c_g * RT
        for side in (cell.ca, cell.an): 
            p_g[side.ix, ...] = p_g[side.ch.ix, ...]
        c_g = p_g / RT 

        x_g_k = cg_k / np.maximum(c_g, 1e-30)[:, np.newaxis, ...]
        x_g_k[:, m.gas_model.i['n2'], ...] = (
            1 
            - x_g_k[:, m.gas_model.i['h2'], ...] 
            - x_g_k[:, m.gas_model.i['h2o'], ...]
            - x_g_k[:, m.gas_model.i['o2'], ...]
        )
        p_g_k = p_g[:, np.newaxis, ...] * x_g_k
        D_g_k = m.gas_diffusion_model.species_diffusion_coefficient(T, p_g, x_h2=self.x_h2_mask)
        M_g   = np.sum(x_g_k * m.gas_model.molecular_weights[np.newaxis, :, np.newaxis], axis=1)
        rho_g = c_g * M_g 

        c_sat = water_saturation_concentration(T)
        c_v   = cg_k[:, -1, ...]
        rh    = c_v / c_sat
        rho_l = water_density(T)
        nu_l  = water_kinematic_viscosity(T)
        nu_g  = m.gas_model.mixture_dynamic_viscosity(T, x_g_k, 
            m.gas_model.molecular_weights) / np.maximum(1e-12, rho_g)
        V_w   = water_molar_volume(T)

        f_v = m.memb_model.water_vol_fraction(lmbd, V_w, cell.V_ion)

        memb_ix  = cell.memb.ix
        ca_cl_ix = cell.ca.cl.ix
        an_cl_ix = cell.an.cl.ix

        return CellState(
            x=x,
            lmbd=lmbd, T=T, cg_k=cg_k, s=s,
            iF=iF, p_g=p_g, p_g_k=p_g_k, D_g_k=D_g_k,
            c_sat=c_sat, c_v=c_v, rh=rh,
            rho_l=rho_l, nu_l=nu_l, M_g=M_g, rho_g=rho_g, nu_g=nu_g, f_v=f_v,
            T_memb=T[memb_ix, ...],
            T_ca_cl=T[ca_cl_ix, ...],
            T_an_cl=T[an_cl_ix, ...],
            f_v_memb=f_v[memb_ix, ...],
            f_v_ca_cl=f_v[ca_cl_ix, ...],
            f_v_an_cl=f_v[an_cl_ix, ...],
            lmbd_ca_cl=lmbd[ca_cl_ix, ...],
            p_h2=p_g_k[an_cl_ix, 2, ...],
            p_o2_ca_cl=p_g_k[ca_cl_ix, 0, ...],
        )

    def _compute_voltage(self, state: CellState):
        """Compute local O₂ pressure and cell voltage; populate state in-place."""
        cell = self.cell
        m    = self.base_model

        state.t_water_film = m.cl_model.water_film_thickness(
            state.s[cell.ca.cl.ix, ...], cell.ca.cl,
        )
        state.R_o2_local = m.cl_model.o2_ionomer_film_resistance(
            state.f_v_ca_cl, state.T_ca_cl, cell.ca.cl, m.memb_model,
            cell.ca.cl.t_ion_film, state.t_water_film, coverage_ratio=0,
        )
        c_o2_local = np.maximum(
            state.cg_k[cell.ca.cl.ix, 0, ...] - state.R_o2_local * state.iF / 4,
            1e-30,
        )
        state.p_o2_local = c_o2_local * ct.gas_constant * state.T_ca_cl

        (state.V_cell, state.eta_ohm, state.eta_act,
         state.E_rev_ca, state.E_rev_an,
         state.eta_memb, _, state.eta_gdl) = m.voltage_model.calculate_cell_voltage(
            T_an_cl=state.T_an_cl,
            T_ca_cl=state.T_ca_cl,
            T_memb=state.T_memb,
            f_v_an_cl=state.f_v_an_cl,
            f_v_memb=state.f_v_memb,
            f_v_ca_cl=state.f_v_ca_cl,
            s_ca_cl=0,
            p_h2=state.p_h2,
            p_o2_local=state.p_o2_local,
            i=state.iF * ct.faraday,
            memb=cell.memb,
            electrical_resistance=cell.electrical_resistance,
            memb_model=m.memb_model,
            ionomer_model=m.memb_model,
            ca_cl_model=m.cl_model,
            ca_cl=cell.ca.cl,
            charge=cell.charge,
        )

    def _compute_resistances(self, state: CellState):
        """Build layer resistance array R and harmonic-mean inter-layer eff_R."""
        cell = self.cell
        m    = self.base_model
        R = np.zeros_like(state.x)

        R[:, self.i_s, ...] = m.darcy_transport_model.calculate_darcy_flow_resistance(
            state.s, state.nu_l, cell.thickness,
            cell.K_abs, cell.n_rel,
        )
        
        R[:, self.i_cg, ...] = m.gas_diffusion_model.total_diffusion_resistance(
            state.T[:, np.newaxis, ...],
            state.s[:, np.newaxis, ...],
            state.D_g_k,
            m.gas_model.molecular_weights[np.newaxis,:,np.newaxis], 
            cell.thickness[:, np.newaxis, ...],
            cell.eps_p[:, np.newaxis, ...],
            cell.tort[:, np.newaxis, ...],
            cell.d_p[:, np.newaxis, ...],
            cell.n_s[:, np.newaxis, ...],
        )
        R[:, self.i_T, ...] = cell.thickness / cell.bulk_thermal_conductivity

        # λ transport is only defined in the ionomer domain; default to inf everywhere
        # and overwrite for ionomer layers only to avoid divide-by-zero on other layers.
        D_lmbd = m.memb_model.diffusion_coefficient(
            state.lmbd, state.T,
            cell.darken_num_ion,
            cell.darken_den_ion,
            cell.D_lmbd_ref_ion,
            cell.E_act_ion,
        )
        R[:, self.i_lmbd, ...] = m.memb_model.calculate_membrane_water_resistance(
            D_lmbd, cell.thickness, cell.eps_ion,
            cell.c_ion, cell.tort_ion,
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
        state.R = R
        state.eff_R = eff_R
        return R, eff_R

    def _compute_fluxes(self, eff_R, state: CellState):
        """Compute inter-layer flux array J and store it on *state*.

        Fluxes are computed by Fick/Fourier/Darcy diffusion plus
        electro-osmotic drag (EOD) across the ionomer interfaces.

        Returns
        -------
        J : ndarray, shape (n_layers+1, n_variables, ...)
        """
        cell = self.cell
        m    = self.base_model
        phi = state.x.copy()

        # Liquid-water driving potential: gas pressure + capillary pressure
        phi[:, self.i_s, ...] = state.p_g
        phi[cell.ca.ix + cell.an.ix, self.i_s, ...] += (
            m.darcy_transport_model.capillary_pressure_from_saturation(
                state.s,
                cell.p_b,
                cell.van_genuchten_m,
                cell.van_genuchten_n,
            )[cell.ca.ix + cell.an.ix, ...]
        )

        J = np.zeros((self.n_layers + 1, self.n_variables, state.x.shape[-1]))
        J[1:-1, ...] = -(phi[1:, ...] - phi[:-1, ...]) / eff_R

        # Electro-osmotic drag: add EOD flux at each CL/membrane interface.
        # J_eod is indexed over [an.cl, memb, ca.cl]; index [0] is used at the
        # an.cl/memb boundary and index [2] at the memb/ca.cl boundary so that
        # each interface uses the catalyst-layer-side drag coefficient.
        J_eod = (
            cell.memb.calculate_electroosmotic_drag_coefficient(state.T, state.lmbd)
            [self.ionomer_domain, ...]
            * state.iF
        )
        J[[cell.an.cl.ix + 1, cell.memb.ix + 1], self.i_lmbd, ...] += J_eod[[0, 2], ...]

        state.J = J
        return J

    def _compute_water_exchange(self, state: CellState):
        """Return ionomer–gas water exchange flux J_des [kmol m⁻² s⁻¹].

        Positive values indicate desorption (water leaving the ionomer into
        the gas phase); negative values indicate absorption.

        The rate coefficient uses the measured desorption value ``k_des_ref_ion``
        (Ge et al. 2005, a_d = 1.42e-2 cm/s) for desorption (λ > λ_eq) and
        the absorption value (a_a = 3.53e-3 cm/s = a_d × 3.53/14.2) for
        absorption (λ < λ_eq), both with an Arrhenius temperature correction.
        """
        cell = self.cell
        m    = self.base_model
        k_des = m.memb_model.sorption_coefficient(
            state.f_v, state.T,
            cell.k_des_ref_ion,
            cell.E_act_ion,
            cell.T_ref_des_ion,
        )
        lmbd_eq = (
            (1 - state.s) * m.memb_model.equilibrium_water_content(
                state.rh, cell.sorption_coeffs_ion,
            )
            + state.s * m.memb_model.liquid_equilibrium_water_content(
                cell.lmbd_liq_ref_ion,
            )
        )
        state.lmbd_eq = lmbd_eq
        delta_lmbd = state.lmbd - lmbd_eq
        # Scale down absorption relative to desorption: ratio a_a/a_d from Ge et al.
        rate_factor = np.where(delta_lmbd > 0, 1.0, 3.53 / 14.2)
        return k_des * delta_lmbd * cell.c_ion * rate_factor

    def _compute_phase_change(self, state: CellState):
        """Return vapour–liquid phase-change source term S_vl.

        Positive when condensation occurs (c_v > c_sat), negative when evaporating.
        """
        state.c_sat  = water_saturation_concentration(state.T)
        factor = np.where(state.c_sat > state.c_v, state.s, 1 - state.s)
        state.S_lv = 1000.0 * (state.c_v - state.c_sat) * factor    
        return state.S_lv
    
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
        S_T_losses[ca_cl_ix,  ...] = (state.eta_act - std_formation_entropy_h2ol / (2 * ct.faraday)) * i
        S_T_losses[an_cl_ix,  ...] = 0
        S_T_losses[ca_gdl_ix, ...] = state.eta_gdl   * i
        S_T_losses[an_gdl_ix, ...] = state.eta_gdl   * i

        # Ionomer water: absorption sink 
        S[cell.an.cl.ix, self.i_lmbd, ...] -= (
            J_des[cell.an.cl.ix, ...]
        ) / cell.an.cl.thickness
        S[cell.ca.cl.ix, self.i_lmbd, ...] -= (
            J_des[cell.ca.cl.ix, ...]
        ) / cell.ca.cl.thickness

        # Liquid water: phase change (blocked in membrane)
        S[:, self.i_s, ...]            = S_vl * water_molecular_weight
        S[cell.memb.ix, self.i_s, ...] = 0

        # Water vapour: condensation/evaporation sink
        S[:, self.i_cg[-1], ...] = -S_vl

        # Ionomer water (λ): electrochemical production at cathode CL.
        # Water is produced in dissolved form at the Pt–ionomer boundary (Vetter & Schumacher
        # 2019, Eq. 6), so it enters the ionomer directly rather than the gas phase.
        S[cell.ca.cl.ix, self.i_lmbd, ...] += state.iF / 2 / cell.ca.cl.thickness

        # Gas species: reactant consumption by electrochemistry
        S[cell.ca.cl.ix, self.i_cg[0], ...] = (-state.iF / 4) / cell.ca.cl.thickness
        S[cell.an.cl.ix, self.i_cg[2], ...] = (-state.iF / 2) / cell.an.cl.thickness

        # Water vapour: ionomer–gas exchange (desorption adds to vapour, absorption removes)
        for side in (cell.an, cell.ca):
            S[side.cl.ix, self.i_cg[-1], ...] += J_des[side.cl.ix, ...] / side.cl.thickness
        S[cell.memb.ix, self.i_cg, ...] = 0

        # Temperature: ohmic/activation losses + phase-change enthalpy
        h_vl = enthalpy_condensation(state.T)
        S[:, self.i_T, ...] = S_T_losses / cell.thickness + S_vl * h_vl
        for side in (cell.an, cell.ca):
            S[side.cl.ix, self.i_T, ...] -= (
                J_des[side.cl.ix, ...] / side.cl.thickness
                * self.base_model.memb_model.heat_of_adsorption(
                    state.T[side.cl.ix, ...], cell.memb)
            )
        state.S = S
        return S

    def _compute_capacities(self, state: CellState):
        """Populate capacity array C."""
        cell = self.cell
        C = np.ones_like(state.x)

        # Membrane lambda is quasi-steady → inf capacity; only CLs have finite dynamics
        C[:, self.i_lmbd, ...] = (
            cell.eps_ion * cell.c_ion
        )
        C[:, self.i_s,  ...]   = state.rho_l * cell.eps_p
        C[:, self.i_cg, ...]   = cell.eps_p[:, np.newaxis, ...]
        C[:, self.i_T, ...]    = cell.bulk_density * cell.bulk_specific_heat_capacity
        np.nan_to_num(C, copy=False, nan=np.inf)
        # Non-ionomer layers (GDL, channel) have eps_ion=0, giving C_lmbd=0.
        # With no λ flux or source there, dxdt=0/0=NaN — block dynamics instead.
        non_ionomer = ~np.array(self.ionomer_domain)
        C[non_ionomer, self.i_lmbd, ...] = np.inf
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

        # 2. Local O₂ pressure + cell voltage
        self._compute_voltage(state)

        # 3. Transport resistances
        _, eff_R = self._compute_resistances(state)

        # 4. Inter-layer fluxes (including EOD)
        J = self._compute_fluxes(eff_R, state)

        # 5. Water exchange between phases
        J_des = self._compute_water_exchange(state)
        S_vl  = self._compute_phase_change(state)

        # 6. Source terms
        S = self._compute_sources(state, J_des, S_vl)

        # 7. Capacities
        C = self._compute_capacities(state)

        # 8. Assemble dxdt; enforce channel boundary conditions
        dxdt = ((J[:-1, ...] - J[1:, ...]) / cell.thickness[:, np.newaxis] + S) / C
        for ch in (cell.ca.ch, cell.an.ch):
            dxdt[ch.ix, self.i_T,  :] = 0
            dxdt[ch.ix, self.i_cg, :] = 0
            dxdt[ch.ix, self.i_s,  :] = 0

        return dxdt.reshape(self.n_layers * self.n_variables, state.x.shape[-1])
