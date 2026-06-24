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
    # sol.y[0]   → T_MEA(t)  [K]
    # sol.y[1:]  → λ(ξ, t)   [mol H2O / mol site]

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

        # Fresh state from current operating conditions
        state = self._ss_model._init_state(
            cell,
            cond.cell_temperature,
            cond.current_density,
            cond.ca,
            cond.an,
        )

        # Thermal quantities from ODE state
        state.thermal_resistance = self.thermal_model.heat_transfer_resistance(cell)
        self.thermal_model.set_mea_temperature(T_mea, cell, state)

        # Water balance — membrane profile prescribed from ODE state
        self.water_balance_model.calculate_water_transport(
            cell, state,
            dynamic=True,
            water_profile=water_profile,
            gas_transport_model=self.gas_transport_model,
        )

        # Cathode liquid saturation (quasi-static: computed from net water flux)
        self.water_balance_model.calculate_water_saturation(cell.ca, state.ca)
        cell.ca.cl.set_water_film_thickness(state.ca.cl.non_wetting_saturation)

        # Update H2O vapour transport resistance with the new saturation
        gtr = self.gas_transport_model
        state.ca.h2ov_transport_resistance = gtr.gas_transport_resistance(cell.ca, state.ca, 'h2o')
        state.an.h2ov_transport_resistance = gtr.gas_transport_resistance(cell.an, state.an, 'h2o')

        # Gas concentrations and cell voltage
        gtr.calculate_gas_concentrations(cell, state)
        self.voltage_model.compute_cell_voltage(cell, state)

        # Rates of change
        dTdt = self.thermal_model.temperature_rate_of_change(cell, state)
        dlambdadt = self.water_balance_model.membrane_water_rate_of_change(
            cell, state, n
        )

        return np.concatenate([[float(dTdt)], np.asarray(dlambdadt).ravel()])

    def solve(self, cell, cell_conditions, t_span, x0=None, **kwargs):
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
            steady-state solve is run automatically.
        **kwargs
            Forwarded to :func:`scipy.integrate.solve_ivp`.  Defaults to
            ``method='Radau'``, ``rtol=1e-4``, ``atol=1e-6``.

        Returns
        -------
        scipy.integrate.OdeResult
            Standard result object.  Key attributes:

            * ``sol.t`` — time points (s)
            * ``sol.y[0]`` — T_MEA(t) [K]
            * ``sol.y[1:]`` — membrane water profile λ(ξ, t)
        """
        from scipy.integrate import solve_ivp

        if x0 is None:
            cond0 = cell_conditions(t_span[0]) if callable(cell_conditions) else cell_conditions
            _, x0 = self.set_initial_conditions(cell, cond0)

        kwargs.setdefault('method', 'BDF')
        kwargs.setdefault('rtol', 1e-4)
        kwargs.setdefault('atol', 1e-6)

        return solve_ivp(
            lambda t, x: self.f_transient(t, x, cell, cell_conditions),
            t_span,
            np.asarray(x0, dtype=float),
            **kwargs,
        )
