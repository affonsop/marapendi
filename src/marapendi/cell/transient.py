"""
Transient PEMFC model: coupled MEA temperature and membrane water content ODE.

:class:`TransientModel` integrates two sets of coupled ODEs:

* **MEA temperature** T(t) — from the lumped thermal energy balance with a
  surface heat capacity ``cell.mea_surface_heat_capacity`` [J/(m²·K)].
* **Membrane water-content profile** λ(ξ, t) — from Fickian diffusion and
  electroosmotic drag, discretised into ``n_memb_mesh`` finite-volume nodes.

The state vector is ``x = [T_MEA, λ_0, …, λ_{n-1}]``.

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
    # sol.y[0]           → T_MEA(t)  [K]
    # sol.y[1:]          → λ(ξ, t)   [mol H2O / mol site]
    # sol.diagnostics    → dict of arrays (voltage, HFR, water contents, …)

    # Or evaluate at custom time points from a dense-output solution:
    diag = model.evaluate(cell, conditions, t_eval, x_eval=sol.sol(t_eval))

References
----------
Ferrara, A. et al. J. Power Sources 390, 197–207 (2018).
Goshtasbi, A. et al. J. Electrochem. Soc. 167, 024518 (2020).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from .thermal import ThermalModel
from .gas_transport import GasTransportModel
from .voltage import VoltageModel
from .explicit_steady_state import ExplicitSteadyStateModel
from ..water_balance.water_balance import WaterBalanceModel
from ..water_balance.membrane_transient import MembraneWaterBalanceTransientModel
from .state import CellState


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
    n_memb_mesh: int = 3

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
            Initial state ``[T_MEA, λ_0, …, λ_{n-1}]``.
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

        x0 = np.concatenate([[T0], lmbd0])
        return state, x0

    def f_transient(self, t, x, cell, cell_conditions) -> np.ndarray:
        """ODE right-hand side for :func:`scipy.integrate.solve_ivp`.

        Parameters
        ----------
        t : float
            Current time (s).
        x : np.ndarray, shape (1 + n_memb_mesh,)
            ``[T_MEA, λ_0, …, λ_{n-1}]``.
        cell : FuelCell
        cell_conditions : CellConditions or callable(t) -> CellConditions

        Returns
        -------
        np.ndarray, shape (1 + n_memb_mesh,)
            ``[dT/dt, dλ_0/dt, …, dλ_{n-1}/dt]``.
        """
        n = self.n_memb_mesh
        T_mea = float(x[0])
        water_profile = x[1:1 + n]

        cond = cell_conditions(t) if callable(cell_conditions) else cell_conditions
        state = self._eval_state(cell, cond, T_mea, water_profile)

        # Rates of change
        dTdt = self.thermal_model.temperature_rate_of_change(cell, state)
        dlambdadt = self.water_balance_model.membrane_water_rate_of_change(
            cell, state, n
        )

        return np.concatenate([[float(dTdt)], np.asarray(dlambdadt).ravel()])

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
            * ``hfr`` — high-frequency resistance (Ω·m²)
            * ``membrane.water_content`` — mean membrane water content
            * ``membrane.water_content_profile`` — shape ``(n_memb_mesh, n_t)``
            * ``ca.cl.ionomer_water_content``, ``an.cl.ionomer_water_content``
            * ``ca.cl.liquid_saturation``, ``ca.cl.proton_resistance``
            * ``ca.water_flux``, ``ca.liquid_flux``, ``ca.membrane_water_flux``,
              ``ca.max_vapor_removal_flux``
        """
        from ..simulation.conditions import SideConditions, CellConditions

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

        T_mea_arr     = x_eval[0, :]    # (n_t,)
        water_profile = x_eval[1:, :]   # (n_memb_mesh, n_t)

        state = self._eval_state(cell, cond_all, T_mea_arr, water_profile)
        state.hfr = self.voltage_model.high_frequency_resistance(cell, state)
        return state

    def solve(self, cell, cell_conditions, t_span, x0=None,
              compute_diagnostics=True, **kwargs):
        """Integrate the transient ODE over *t_span*.

        Parameters
        ----------
        cell : FuelCell
        cell_conditions : CellConditions or callable(t) -> CellConditions
            Constant or time-varying operating conditions.
        t_span : tuple of float
            ``(t_start, t_end)`` in seconds.
        x0 : array_like, optional
            Initial state ``[T_MEA, λ_0, …, λ_{n-1}]``.  When omitted a
            steady-state solve is run automatically via
            :meth:`set_initial_conditions`.
        compute_diagnostics : bool, optional
            When ``True`` (default), :meth:`evaluate` is called at the
            solver's internal time steps after the ODE integration completes
            and the result is stored as ``sol.diagnostics``.  Set to ``False``
            to skip the post-processing step (e.g. when only ODE trajectories
            are needed).
        **kwargs
            Forwarded to :func:`scipy.integrate.solve_ivp`.  Defaults:
            ``method='Radau'``, ``rtol=1e-4``, ``atol=1e-6``.

        Returns
        -------
        scipy.integrate.OdeResult
            Standard result object extended with:

            * ``sol.t`` — time points (s)
            * ``sol.y[0]`` — T_MEA(t) [K]
            * ``sol.y[1:]`` — membrane water profile λ(ξ, t)
            * ``sol.diagnostics`` — :class:`~marapendi.cell.state.CellState`
              from :meth:`evaluate` at ``sol.t``
              (only present when ``compute_diagnostics=True``).
        """
        from scipy.integrate import solve_ivp

        if x0 is None:
            cond0 = cell_conditions(t_span[0]) if callable(cell_conditions) else cell_conditions
            _, x0 = self.set_initial_conditions(cell, cond0)

        kwargs.setdefault('method', 'Radau')
        kwargs.setdefault('rtol', 1e-4)
        kwargs.setdefault('atol', 1e-6)

        sol = solve_ivp(
            lambda t, x: self.f_transient(t, x, cell, cell_conditions),
            t_span,
            np.asarray(x0, dtype=float),
            **kwargs,
        )

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

        return state
