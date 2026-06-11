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
)


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
    conditions: object = field(default=None, repr=False, compare=False)  # optional CellConditions
    # Injected by CellBaseModel.__post_init__; never set by the caller.
    base_model: object = field(default=None, repr=False, compare=False)

    def get_inputs(self, t: float) -> dict:
        """Return inputs dict evaluated at time *t*."""
        i = self.current_density(t) if callable(self.current_density) else float(self.current_density)
        result = {'i': i}
        if self.conditions is not None:
            result['conditions_snapshot'] = self.conditions.at(t)
        return result

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

        self._rebuild_mask()

    def _rebuild_mask(self) -> None:
        """Compute the active-state mask for the current conditions setting.

        States that are structurally always dxdt=0 (no ionomer layer for λ,
        channel nodes for T and s, membrane for gas, wrong side for O₂/H₂, …)
        are excluded from the ODE.  The reduced state that scipy sees has
        shape ``(n_active,)`` instead of ``(n_layers × n_variables,)``.

        Uses position-based logic (enumerate index) rather than layer.ix so that
        models sharing the same layer object across sides do not cause index
        corruption.  Call again if ``self.conditions`` is changed after construction.
        """
        cell = self.cell
        channel_gas_active = self.conditions is not None

        # Locate key layers by Python identity (unique objects, safe against sharing)
        an_ch_pos = next(i for i, l in enumerate(cell.layers) if l is cell.an.ch)
        ca_ch_pos = next(i for i, l in enumerate(cell.layers) if l is cell.ca.ch)
        memb_pos  = next(i for i, l in enumerate(cell.layers) if l is cell.memb)
        an_cl_pos = next(i for i, l in enumerate(cell.layers) if l is cell.an.cl)
        ca_cl_pos = next(i for i, l in enumerate(cell.layers) if l is cell.ca.cl)

        # Side membership by position (anode < membrane < cathode in the canonical ordering)
        ch_pos  = {an_ch_pos, ca_ch_pos}
        an_pos  = set(range(0, memb_pos))                     # an.ch … an.cl
        ca_pos  = set(range(memb_pos + 1, self.n_layers))      # ca.cl … ca.ch
        ion_pos = {an_cl_pos, memb_pos, ca_cl_pos}

        mask = np.zeros((self.n_layers, self.n_variables), dtype=bool)

        for i in range(self.n_layers):
            is_ch   = i in ch_pos
            is_an   = i in an_pos
            is_ca   = i in ca_pos
            is_memb = i == memb_pos
            gas_ok  = not is_ch or channel_gas_active

            # λ — ionomer layers only
            mask[i, self.i_lmbd]   = i in ion_pos
            # T — everywhere except channel nodes (frozen to boundary condition)
            mask[i, self.i_T]      = not is_ch
            # O₂ — cathode porous + cathode channel if channel dynamics
            mask[i, self.i_cg[0]] = is_ca and not is_memb and gas_ok
            # N₂ — channel nodes only when channel dynamics; derived from pressure balance in porous
            mask[i, self.i_cg[1]] = is_ch and channel_gas_active
            # H₂ — anode porous + anode channel if channel dynamics
            mask[i, self.i_cg[2]] = is_an and not is_memb and gas_ok
            # H₂O — all non-membrane layers; channel only if channel dynamics
            mask[i, self.i_cg[3]] = not is_memb and gas_ok
            # s — porous layers only (not membrane, not channel)
            mask[i, self.i_s]      = not is_memb and not is_ch

        flat               = mask.flatten()
        self._active_mask  = flat
        self._active_ix    = np.where(flat)[0]
        self._inactive_ix  = np.where(~flat)[0]
        self._n_active     = int(flat.sum())
        # Background vector holds frozen normalized state values.
        # Zeros by default; properly set once initial_state() is called.
        self._background = np.zeros(self.n_layers * self.n_variables)

    def expand_state(self, y_active: np.ndarray) -> np.ndarray:
        """Reconstruct the full ``(n_layers × n_variables, m)`` state from the
        active (compressed) portion by inserting frozen background values."""
        if y_active.ndim == 1:
            y_full = self._background.copy()
            y_full[self._active_ix] = y_active
            return y_full
        m      = y_active.shape[1]
        y_full = np.tile(self._background[:, np.newaxis], m)
        y_full[self._active_ix] = y_active
        return y_full

    @property
    def n_states(self) -> int:
        """Number of active (non-frozen) ODE states passed to the solver."""
        return self._n_active

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

        y_full = (x0 / self.norm_factor).flatten()
        self._background = y_full.copy()   # frozen values for inactive states
        return y_full[self._active_mask]   # only active states passed to solver

    def initial_state_from_conditions(
        self,
        snapshot,
        s_ca: float = 0.,
        s_an: float = 0.,
    ) -> np.ndarray:
        """Build initial state from a CellSnapshot (differential-cell assumption).

        Channel concentrations are set to inlet mole fractions at backpressure,
        valid when stoichiometry ≫ 1 so that gas depletion along the channel
        is negligible.

        Parameters
        ----------
        snapshot : CellSnapshot
            Frozen snapshot at t=0 from ``CellConditions.at(0.)``.
        s_ca, s_an : float
            Initial liquid saturation at cathode / anode sides.
        """
        cell = self.cell
        ca, an = snapshot.ca, snapshot.an

        def _concentrations(cond):
            x_k = cond.inlet_gas_molar_flow_rates / cond.inlet_gas_molar_flow_rate
            return x_k * cond.backpressure / (ct.gas_constant * cond.temperature)

        c_ca = _concentrations(ca)
        c_an = _concentrations(an)

        rh_ca = c_ca[3] / water_saturation_concentration(ca.temperature)
        rh_an = c_an[3] / water_saturation_concentration(an.temperature)

        x0 = np.zeros((self.n_layers, self.n_variables))
        rh_mean = np.full(self.n_layers, (rh_ca + rh_an) / 2)
        lmbd_eq = self.base_model.memb_model.equilibrium_water_content(
            rh_mean, cell.sorption_coeffs_ion,
        )[:, 0]
        x0[:, self.i_lmbd] = np.nan_to_num(lmbd_eq, nan=0.)
        x0[:, self.i_T] = (ca.temperature + an.temperature) / 2
        x0[cell.ca.ix, self.i_s] = s_ca
        x0[cell.an.ix, self.i_s] = s_an

        memb_ix = cell.memb.ix
        for layer in cell.layers:
            if layer is cell.memb:
                continue
            x0[layer.ix, self.i_cg] = c_ca if layer.ix > memb_ix else c_an

        y_full = (x0 / self.norm_factor).flatten()
        self._background = y_full.copy()   # frozen values for inactive states
        return y_full[self._active_mask]   # only active states passed to solver

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
        T    = np.minimum(np.maximum(x[:, self.i_T, ...], 275.), 600.)
        cg_k = np.maximum(x[:, self.i_cg, ...], 0.)
        s    = np.minimum(np.maximum(x[:, self.i_s, ...], 0.), 1.)
        return lmbd, T, cg_k, s

    # ------------------------------------------------------------------
    # rates_of_change — sub-steps
    # ------------------------------------------------------------------

    def _compute_derived_quantities(self, x, i) -> CellState:
        """Unpack state vector; compute water/ionomer properties; delegate gas state to GasMixtureModel."""
        cell = self.cell
        m    = self.base_model
        lmbd, T, cg_k, s = self.get_states_from_x(x)

        
        memb_ix  = cell.memb.ix
        ca_cl_ix = cell.ca.cl.ix
        an_cl_ix = cell.an.cl.ix

        csat = water_saturation_concentration(T)

        

        rho_l = water_density(T)
        nu_l  = water_kinematic_viscosity(T)

        iF  = i / ct.faraday
        V_w = water_molar_volume(T)
        f_v = m.memb_model.water_vol_fraction(lmbd, V_w, cell.V_ion)


        state = CellState(
            x=x,
            lmbd=lmbd, T=T, cg_k=cg_k, s=s, iF=iF,
            c_sat=csat,
            c_v=cg_k[:, -1, ...],
            rh=cg_k[:, -1, ...] / csat,
            rho_l=rho_l,
            nu_l=nu_l,
            mu_l=nu_l * rho_l,
            f_v=f_v,
            T_memb=T[memb_ix, ...],
            T_ca_cl=T[ca_cl_ix, ...],
            T_an_cl=T[an_cl_ix, ...],
            f_v_memb=f_v[memb_ix, ...],
            f_v_ca_cl=f_v[ca_cl_ix, ...],
            f_v_an_cl=f_v[an_cl_ix, ...],
            lmbd_ca_cl=lmbd[ca_cl_ix, ...],
        )

        # Gas mixture derived quantities (p_g, x_g_k, D_g_k, M_g, rho_g, nu_g, mu_g, ...)

        return state

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def rates_of_change(self, x, i=0., conditions_snapshot=None):
        """Return dxdt for all state variables across all layers.

        Parameters
        ----------
        conditions_snapshot : CellSnapshot, optional
            When provided, channel gas concentrations are solved dynamically:
            inlet/outlet flows are added as source terms and dxdt[ch, cg] is
            not forced to zero.  N₂ through-plane transport is disabled
            (R_N₂ = ∞ in porous layers); its concentration is derived from
            the pressure balance by GasMixtureModel.
        """
        cell = self.cell
        bm   = self.base_model

        # Expand compressed active state → full (n_layers × n_variables, m) state.
        # BaseModel always passes 2D (n_active, m) here.
        x_full = self.expand_state(x)    # (n_full, m)
        x = (
            x_full.reshape(self.n_layers, self.n_variables, x_full.shape[1])
            * self.norm_factor[..., np.newaxis]
        )

        # 1. Derived thermodynamic quantities
        state = self._compute_derived_quantities(x, i)
        bm.gas_model.compute_state(state, cell, self, bm.gas_diffusion_model)
        bm.cl_model.compute_local_o2_partial_pressure(state, cell, bm.memb_model)
        bm.voltage_model.compute_cell_voltage(state, cell, bm.memb_model, bm.cl_model)

        # 3. Initialise transport matrices
        m = x.shape[-1]
        state.C   = np.ones((self.n_layers, self.n_variables, m))
        state.R   = np.full((self.n_layers, self.n_variables, m), np.inf)
        state.S   = np.zeros((self.n_layers, self.n_variables, m))
        state.phi = x.copy()
        state.J   = np.zeros((self.n_layers + 1, self.n_variables, m))
        state.J_des = None
        state.S_lv  = None

        # 4. Delegate matrix filling to sub-models
        #    Order: membrane (sets J_des) → darcy (sets S_lv) → gas → thermal
        bm.memb_model.update_transport_matrices(state, cell, self)
        bm.darcy_transport_model.update_transport_matrices(state, cell, self)
        bm.gas_diffusion_model.update_transport_matrices(state, cell, self, bm.gas_model)
        bm.thermal_model.update_transport_matrices(state, cell, self, bm.memb_model)

        # 4b. Channel gas dynamics: inlet/outlet source terms + no N₂ through-plane flux
        if conditions_snapshot is not None:
            i_N2 = self.i_cg[1]
            # Non-channel N₂ is masked out of the ODE (always dxdt=0) so we only
            # need to block through-plane transport at the porous layers for consistency.
            for layer in cell.layers:
                if layer not in (cell.ca.ch, cell.an.ch):
                    state.R[layer.ix, i_N2, ...] = np.inf
            for side, cond in (
                (cell.ca, conditions_snapshot.ca),
                (cell.an, conditions_snapshot.an),
            ):
                bm.flowfield_model.add_channel_inlet_outlet_flows(cond, state, side.ch, self)

        mask = np.isnan(state.R)
        if mask.any():
            state.R[mask] = np.inf

        # 5. Effective inter-layer resistance
        eff_R = (state.R[:-1] + state.R[1:]) / 2
        eff_R[-1, self.i_T] += cell.thermal_resistance / 2
        eff_R[ 0, self.i_T] += cell.thermal_resistance / 2
        mask = np.isnan(eff_R)
        if mask.any():
            eff_R[mask] = np.inf
        state.eff_R = eff_R

        # 6. Bulk inter-layer fluxes
        state.J[1:-1] = -(state.phi[1:] - state.phi[:-1]) / eff_R

        # 7. Flux corrections (EOD adds to λ flux after bulk diffusion is computed)
        bm.memb_model.add_eod_flux(state, cell, self)

        # 8. Assemble dxdt and return only active states.
        # Inactive states are not passed to the solver, so no explicit zeroing needed —
        # the mask guarantees they never appear in the ODE output.
        mask = np.isnan(state.C)
        if mask.any():
            state.C[mask] = np.inf
        dxdt = (
            (state.J[:-1] - state.J[1:]) / cell.thickness[:, np.newaxis] + state.S
        ) / state.C

        dxdt_norm = (dxdt / self.norm_factor[..., np.newaxis]).reshape(
            self.n_layers * self.n_variables, state.x.shape[-1]
        )
        return dxdt_norm[self._active_ix]   # compress to active states only
