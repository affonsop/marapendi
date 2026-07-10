"""
Transient PEMFC model, with coupled MEA temperature and membrane water content ODE.

:class:`TransientModel` integrates two sets of coupled ODEs:

* **MEA temperature** — from the lumped thermal energy balance with a
  surface heat capacity ``cell.mea_surface_heat_capacity`` (J/(m²·K)).
* **Membrane water-content profile** — from Fickian diffusion and
  electroosmotic drag, discretised into ``n_memb_mesh`` finite-volume nodes.

The ODE state vector has ``1 + n_memb_mesh`` entries: the MEA temperature
followed by the water content at each membrane node.

At each timestep the model:

1. Reinitialises gas compositions and flow rates from the current operating
   conditions (which may vary in time via a callable).
2. Overrides the MEA temperature from *x* and computes the thermal resistance.
3. Solves the membrane water balance using the **prescribed** water-content
   profile (transient mode), obtaining boundary absorption/desorption fluxes.
4. Recomputes cathode liquid saturation quasi-statically from the net water flux.
5. Calculates gas concentrations and cell voltage.
6. Returns dT/dt and dλ/dt at each node.

Usage
-----
::

    model = TransientModel(n_memb_mesh=5)
    state, x0 = model.set_initial_conditions(cell, conditions)
    sol = model.solve(cell, conditions, t_span=(0, 3600))
    # sol.y[0]           → MEA temperature (K)
    # sol.y[1:]          → membrane water-content profile
    # sol.diagnostics    → CellState with array-valued fields

    # Or evaluate at custom time points from a dense-output solution:
    diag = model.evaluate(cell, conditions, t_eval, x_eval=sol.sol(t_eval))

References
----------
Ferrara, A. et al. J. Power Sources 390, 197–207 (2018).
Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
"""
from __future__ import annotations

import numpy as np
import types
from dataclasses import dataclass, field

from ..thermal import ThermalModel
from ..gas_transport_resistance import GasTransportModel
from ..voltage import VoltageModel
from .explicit_steady_state import ExplicitSteadyStateModel
from ..water_balance.water_balance import WaterBalanceModel
from ..water_balance.membrane_transient import MembraneWaterBalanceTransientModel
from ...simulation.state import CellState


class _PiecewiseDenseOutput:
    """Dense-output callable stitching together per-segment ``OdeSolution`` interpolants."""

    def __init__(self, segments):
        self.t_min = segments[0].t[0]
        self.t_max = segments[-1].t[-1]
        self._segment_ends = np.array([seg.t[-1] for seg in segments])
        self._interpolants = [seg.sol for seg in segments]

    def __call__(self, t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        scalar = np.ndim(t) == 0
        idx = np.clip(np.searchsorted(self._segment_ends, t_arr, side='left'),
                      0, len(self._interpolants) - 1)
        n_states = self._interpolants[0](self.t_min).shape[0]
        out = np.empty((n_states, t_arr.size))
        for i in np.unique(idx):
            mask = idx == i
            out[:, mask] = self._interpolants[i](t_arr[mask])
        return out[:, 0] if scalar else out


def _stitch_solutions(segments):
    """Merge consecutive :func:`~scipy.integrate.solve_ivp` results from :meth:`TransientModel.solve`'s
    per-breakpoint sub-intervals into a single result with the same shape as a one-shot solve."""
    t = np.concatenate([segments[0].t] + [seg.t[1:] for seg in segments[1:]])
    y = np.concatenate([segments[0].y] + [seg.y[:, 1:] for seg in segments[1:]], axis=1)
    success = all(seg.success for seg in segments)
    failed = next((seg for seg in segments if not seg.success), None)

    return types.SimpleNamespace(
        t=t, y=y, success=success,
        status=failed.status if failed is not None else segments[-1].status,
        message=failed.message if failed is not None else segments[-1].message,
        nfev=sum(seg.nfev for seg in segments),
        njev=sum(seg.njev for seg in segments),
        nlu=sum(seg.nlu for seg in segments),
        sol=_PiecewiseDenseOutput(segments) if all(seg.sol is not None for seg in segments) else None,
    )


@dataclass
class TransientModel:
    """Transient PEMFC model integrating MEA temperature and membrane water content.

    Parameters
    ----------
    voltage_model : VoltageModel
    thermal_model : ThermalModel
    gas_transport_model : GasTransportModel
    n_memb_mesh : int
        Number of finite-volume nodes for the membrane water-content profile.
        The ODE state vector has ``1 + n_memb_mesh`` entries.
    """

    voltage_model: VoltageModel = field(default_factory=VoltageModel)
    thermal_model: ThermalModel = field(default_factory=ThermalModel)
    gas_transport_model: GasTransportModel = field(default_factory=GasTransportModel)
    n_memb_mesh: int = 10

    def __post_init__(self):
        self.water_balance_model = WaterBalanceModel(
            membrane_water_balance_model=MembraneWaterBalanceTransientModel(
                n_profile_points=self.n_memb_mesh
            )
        )
        # Steady-state model used for state initialisation (uses default WBM)
        self._ss_model = ExplicitSteadyStateModel(
            self.voltage_model,
            self.thermal_model,
            WaterBalanceModel(),
            self.gas_transport_model,
        )

        self.norm_factors = [353.15, 10.]
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_initial_conditions(self, cell, cell_conditions) -> tuple[CellState, np.ndarray]:
        """Run a steady-state solve and return the initial ODE state vector.

        Parameters
        ----------
        cell : FuelCell
        cell_conditions : CellConditions
            Initial (or constant) operating conditions.

        Returns
        -------
        state : CellState
            Fully populated steady-state solution used as the starting point.
        x0 : np.ndarray, shape (1 + n_memb_mesh,)
            Initial ODE state: MEA temperature followed by the membrane
            water-content profile at each finite-volume node.
        """
        state = self._ss_model.set_initial_conditions(cell, cell_conditions)
        state = self._ss_model.solve(cell, cell_conditions, state)

        T0 = float(np.atleast_1d(state.mea_temperature)[0])

        # Interpolate the steady-state profile to n_memb_mesh nodes
        lmbd_ss = state.membrane.water_content_profile  # (n_ss, ...) or (n_ss,)
        n_ss = lmbd_ss.shape[0]
        lmbd_ss_1d = lmbd_ss[:, 0] if lmbd_ss.ndim > 1 else lmbd_ss
        xi_ss = np.linspace(0, 1, n_ss)
        xi_tr = np.linspace(0, 1, self.n_memb_mesh)
        lmbd0 = np.interp(xi_tr, xi_ss, lmbd_ss_1d)

        x0 = np.concatenate([[T0 / self.norm_factors[0]], lmbd0 / self.norm_factors[1]])
        return state, x0

    def f_transient(self, t, x, cell, cell_conditions, return_state=False) -> np.ndarray:
        """ODE right-hand side for :func:`scipy.integrate.solve_ivp`.

        Parameters
        ----------
        t : float
            Current time (s).
        x : np.ndarray, shape (1 + n_memb_mesh,)
            ODE state: MEA temperature followed by membrane water-content
            at each finite-volume node.
        cell : FuelCell
        cell_conditions : CellConditions or callable(t) -> CellConditions
        return_state : bool, optional
            When ``True``, also return the :class:`~marapendi.simulation.state.CellState`
            computed internally for this ``(t, x)`` — the same one
            :meth:`evaluate` would recompute from scratch for the same point.
            Useful for callers (e.g. a Simulink S-Function) that need
            diagnostics at every integration step and want to avoid a
            second, redundant physics pass; :func:`scipy.integrate.solve_ivp`
            itself has no hook for this, which is why :meth:`solve` still
            calls :meth:`evaluate` separately after integration.

        Returns
        -------
        np.ndarray, shape (1 + n_memb_mesh,)
            Rate of change of MEA temperature followed by rate of change of
            water content at each membrane node.
        state : CellState, optional
            Only returned when ``return_state=True``, as ``(dxdt, state)``.
        """
        n = self.n_memb_mesh
        T_mea = float(x[0] * self.norm_factors[0])
        water_profile = x[1:1 + n] * self.norm_factors[1]

        cond = cell_conditions(t) if callable(cell_conditions) else cell_conditions
        state = self._eval_state(cell, cond, T_mea, water_profile)

        # Rates of change
        dTdt = self.thermal_model.temperature_rate_of_change(cell, state)
        dlambdadt = self.water_balance_model.membrane_water_rate_of_change(
            cell, state, n
        )

        dxdt = np.concatenate([[float(dTdt) / self.norm_factors[0]], np.asarray(dlambdadt / self.norm_factors[1]).ravel()])

        if return_state:
            # Matches what evaluate() adds on top of _eval_state for the same point.
            state.hfr = self.voltage_model.high_frequency_resistance(cell, state)
            return dxdt, state

        return dxdt

    def evaluate(self, cell, cell_conditions, t_eval, x_eval) -> CellState:
        """Compute model diagnostics from ODE states at given time points.

        Runs the full physics pipeline (water balance, gas transport, voltage)
        once in vectorised form for all *n_t* time points and returns a
        :class:`~marapendi.cell.state.CellState` whose array-valued fields
        each have length *n_t* — matching the API of
        :meth:`~marapendi.cell.ExplicitSteadyStateModel.solve`.

        Parameters
        ----------
        cell : FuelCell
        cell_conditions : CellConditions or callable(t) -> CellConditions
        t_eval : array_like, shape (n_t,)
            Time points at which to evaluate (s).
        x_eval : array_like, shape (1 + n_memb_mesh, n_t)
            ODE state ``[T_MEA; λ_profile]`` at each time.

        Returns
        -------
        CellState
            State with array-valued fields of length *n_t*.  Key attributes:

            * ``cell_voltage`` — cell voltage (V)
            * ``mea_temperature`` — MEA temperature (K)
            * ``hfr`` — high-frequency resistance (Ohm·m²)
            * ``membrane.water_content`` — mean membrane water content
            * ``membrane.water_content_profile`` — shape ``(n_memb_mesh, n_t)``
            * ``ca.cl.ionomer_water_content``, ``an.cl.ionomer_water_content``
            * ``ca.cl.liquid_saturation``, ``ca.cl.proton_resistance``
            * ``ca.water_flux``, ``ca.liquid_flux``, ``ca.membrane_water_flux``,
              ``ca.max_vapor_removal_flux``
        """
        from ...simulation.conditions import SideConditions, CellConditions

        t_eval = np.asarray(t_eval, dtype=float)
        x_eval = np.asarray(x_eval, dtype=float)

        # Gather per-time-point conditions and stack into arrays
        if callable(cell_conditions):
            cond_list = [cell_conditions(t_k) for t_k in t_eval]
        else:
            cond_list = [cell_conditions] * len(t_eval)

        def _stack(sides, attr):
            return np.array([float(np.atleast_1d(getattr(s, attr))[0]) for s in sides])

        ca_sides = [c.ca for c in cond_list]
        an_sides = [c.an for c in cond_list]

        cond_all = CellConditions(
            current_density=np.array(
                [float(np.atleast_1d(c.current_density)[0]) for c in cond_list]
            ),
            cell_temperature=np.array(
                [float(np.atleast_1d(c.cell_temperature)[0]) for c in cond_list]
            ),
            ca=SideConditions(
                inlet_temperature=_stack(ca_sides, 'inlet_temperature'),
                inlet_pressure=_stack(ca_sides, 'inlet_pressure'),
                outlet_pressure=_stack(ca_sides, 'outlet_pressure'),
                dry_o2_mole_fraction=_stack(ca_sides, 'dry_o2_mole_fraction'),
                dry_h2_mole_fraction=_stack(ca_sides, 'dry_h2_mole_fraction'),
                inlet_relative_humidity=_stack(ca_sides, 'inlet_relative_humidity'),
                stoichiometry=_stack(ca_sides, 'stoichiometry'),
                inlet_liquid_saturation=_stack(ca_sides, 'inlet_liquid_saturation'),
                inlet_liquid=ca_sides[0].inlet_liquid,
                inlet_liquid_flow_rate=_stack(ca_sides, 'inlet_liquid_flow_rate'),
                inlet_gas_flow_rate=_stack(ca_sides, 'inlet_gas_flow_rate'),
            ),
            an=SideConditions(
                inlet_temperature=_stack(an_sides, 'inlet_temperature'),
                inlet_pressure=_stack(an_sides, 'inlet_pressure'),
                outlet_pressure=_stack(an_sides, 'outlet_pressure'),
                dry_o2_mole_fraction=_stack(an_sides, 'dry_o2_mole_fraction'),
                dry_h2_mole_fraction=_stack(an_sides, 'dry_h2_mole_fraction'),
                inlet_relative_humidity=_stack(an_sides, 'inlet_relative_humidity'),
                stoichiometry=_stack(an_sides, 'stoichiometry'),
                inlet_liquid_saturation=_stack(an_sides, 'inlet_liquid_saturation'),
                inlet_liquid=an_sides[0].inlet_liquid,
                inlet_liquid_flow_rate=_stack(an_sides, 'inlet_liquid_flow_rate'),
                inlet_gas_flow_rate=_stack(an_sides, 'inlet_gas_flow_rate'),
            ),
        )

        T_mea_arr     = x_eval[0, :] * self.norm_factors[0]   # (n_t,)
        water_profile = x_eval[1:, :] * self.norm_factors[1]  # (n_memb_mesh, n_t)

        state = self._eval_state(cell, cond_all, T_mea_arr, water_profile)
        state.hfr = self.voltage_model.high_frequency_resistance(cell, state)
        # Sets state.heat_release as a side effect (dT/dt itself is not used here).
        self.thermal_model.temperature_rate_of_change(cell, state)
        return state

    def solve(self, cell, cell_conditions, t_span, x0=None,
              compute_diagnostics=True, breakpoints=None, **kwargs):
        """Integrate the transient ODE over *t_span*.

        Parameters
        ----------
        cell : FuelCell
        cell_conditions : CellConditions or callable(t) -> CellConditions
            Constant or time-varying operating conditions.
        t_span : tuple of float
            ``(t_start, t_end)`` in seconds.
        x0 : array_like, optional
            Initial ODE state as returned by :meth:`set_initial_conditions`.
            When omitted, a steady-state solve is run automatically.
        compute_diagnostics : bool, optional
            When ``True`` (default), :meth:`evaluate` is called at the
            solver's internal time steps after the ODE integration completes
            and the result is stored as ``sol.diagnostics``.  Set to ``False``
            to skip the post-processing step (e.g. when only ODE trajectories
            are needed).
        breakpoints : array_like, optional
            Interior times where *cell_conditions* is non-smooth (e.g. a
            load-cycle step change).  The solver's local error estimate is
            unreliable across such a kink, so the integration is split into
            one :func:`~scipy.integrate.solve_ivp` call per sub-interval,
            restarting cleanly at each breakpoint instead of risking a step
            that straddles it undetected.  When omitted (default), these are
            taken from ``cell_conditions.discontinuity_times()`` if that
            method exists (e.g. :class:`~marapendi.simulation.load_cycles.LoadCycle`);
            pass ``breakpoints=[]`` to disable this.
        **kwargs
            Forwarded to :func:`scipy.integrate.solve_ivp`.  Defaults:
            ``method='BDF'``, ``max_step=10``, ``rtol=1e-3``, ``atol=1e-5``.

        Returns
        -------
        scipy.integrate.OdeResult
            Standard result object extended with:

            * ``sol.t`` — time points (s)
            * ``sol.y[0]`` — MEA temperature (K)
            * ``sol.y[1:]`` — membrane water-content profile
            * ``sol.diagnostics`` — :class:`~marapendi.cell.state.CellState`
              from :meth:`evaluate` at ``sol.t``
              (only present when ``compute_diagnostics=True``).
        """
        from scipy.integrate import solve_ivp

        if x0 is None:
            cond0 = cell_conditions(t_span[0]) if callable(cell_conditions) else cell_conditions
            _, x0 = self.set_initial_conditions(cell, cond0)

        kwargs.setdefault('method', 'BDF')
        kwargs.setdefault('max_step', 10)
        kwargs.setdefault('rtol', 1e-3)
        kwargs.setdefault('atol', 1e-5)

        if breakpoints is None:
            get_times = getattr(cell_conditions, 'discontinuity_times', None)
            breakpoints = get_times() if callable(get_times) else []

        t_start, t_end = t_span
        edges = [t_start, *sorted(t for t in breakpoints if t_start < t < t_end), t_end]

        fun = lambda t, x: self.f_transient(t, x, cell, cell_conditions)
        x_cur = np.asarray(x0, dtype=float)
        segments = []
        for seg_start, seg_end in zip(edges[:-1], edges[1:]):
            segment = solve_ivp(fun, (seg_start, seg_end), x_cur, **kwargs)
            segments.append(segment)
            if not segment.success:
                break
            x_cur = segment.y[:, -1]

        sol = segments[0] if len(segments) == 1 else _stitch_solutions(segments)

        if compute_diagnostics:
            sol.diagnostics = self.evaluate(cell, cell_conditions, sol.t, x_eval=sol.y)

        return sol

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_state(self, cell, cond, T_mea: float, water_profile) -> CellState:
        """Run the full physics pipeline for one time point.

        Builds a fresh :class:`CellState`, prescribes the MEA temperature and
        membrane water-content profile, and runs water balance → gas transport
        → voltage.  Called by both :meth:`f_transient` and :meth:`evaluate`.

        Parameters
        ----------
        cell : FuelCell
        cond : CellConditions
        T_mea : float
            MEA temperature (K).
        water_profile : array_like, shape (n_memb_mesh,)
            Membrane water-content profile to prescribe.

        Returns
        -------
        CellState
            Populated state (``cell_voltage``, ``membrane.water_content``,
            ``ca.cl.ionomer_water_content``, … are all set).
        """
        state = self._ss_model._init_state(
            cell,
            cond.cell_temperature,
            cond.current_density,
            cond.ca,
            cond.an,
        )
        state.thermal_resistance = self.thermal_model.heat_transfer_resistance(cell)
        self.thermal_model.set_mea_temperature(T_mea, cell, state)

        self.water_balance_model.calculate_water_transport(
            cell, state,
            dynamic=True,
            water_profile=water_profile,
            gas_transport_model=self.gas_transport_model,
        )
        self.water_balance_model.calculate_water_saturation(cell.ca, state.ca)
        cell.ca.cl.set_water_film_thickness(state.ca.cl.non_wetting_saturation)

        gtr = self.gas_transport_model
        state.ca.h2ov_transport_resistance = gtr.gas_transport_resistance(cell.ca, state.ca, 'h2o')
        state.an.h2ov_transport_resistance = gtr.gas_transport_resistance(cell.an, state.an, 'h2o')
        gtr.calculate_gas_concentrations(cell, state)
        self.voltage_model.compute_cell_voltage(cell, state)

        self._ss_model.set_gas_flow_states(cell, cell.ca, state.ca, cond.ca, cond.cell_temperature)
        self._ss_model.set_gas_flow_states(cell, cell.an, state.an, cond.an, cond.cell_temperature)

        return state
