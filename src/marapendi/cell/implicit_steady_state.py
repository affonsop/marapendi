"""
Cell model: implicit steady-state PEMFC performance model.

:class:`ImplicitSteadyStateModel` extends :class:`ExplicitSteadyStateModel`
by solving for the MEA temperature self-consistently: instead of estimating
T_MEA from an explicit formula, it iterates until the heat released by the
cell matches the heat conducted through the GDL/MPL stack.

The fixed-point equation is:

    T_MEA = T_stack + R_th * q(T_MEA)

where ``q`` is the volumetric heat release rate (W/m²), which depends on the
cell voltage and therefore on T_MEA itself.  :func:`scipy.optimize.root` is
used to solve the scalar (or vectorised) residual.

Warm start
----------
After each successful solve, the converged MEA temperature is stored in
:attr:`_last_mea_temperature`.  On the next call, this value is reused as the
initial guess, making sequential evaluations (e.g. time-series sweeps)
significantly faster.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import root

from .explicit_steady_state import ExplicitSteadyStateModel
from .voltage import VoltageModel
from .thermal import ThermalModel
from .water_balance import MembraneWaterBalanceModel
from .gas_transport import GasTransportModel
from .state import CellState


@dataclass
class ImplicitSteadyStateModel(ExplicitSteadyStateModel):
    """
    Implicit steady-state model for PEMFC polarization curves.

    The MEA temperature is found by solving the fixed-point equation

        T_MEA = T_stack + R_th · q(T_MEA, V(T_MEA))

    self-consistently via a nonlinear root-find.  All other physics
    (water balance, gas transport, voltage) are identical to
    :class:`ExplicitSteadyStateModel`.

    Usage
    -----
    ::

        model = ImplicitSteadyStateModel()
        conditions = CellConditions(
            current_density=np.linspace(1e3, 2e4, 20),
            cell_temperature=353.15,
            ca=SideConditions(outlet_pressure=1.5e5, dry_o2_mole_fraction=0.21, ...),
            an=SideConditions(outlet_pressure=1.5e5, dry_h2_mole_fraction=1.0, ...),
        )
        state = model.set_initial_conditions(cell, conditions)
        state = model.solve(cell, conditions, state)

    Parameters
    ----------
    root_kwargs : dict
        Extra keyword arguments forwarded to :func:`scipy.optimize.root`
        (e.g. ``{"method": "hybr", "tol": 1e-6}``).

    Attributes
    ----------
    _last_mea_temperature : ndarray or None
        Cached MEA temperature from the previous :meth:`solve` call, used
        as warm-start initial guess.
    """

    root_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self._last_mea_temperature = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, cell, cell_conditions, initial_state: CellState) -> CellState:
        """Solve for the self-consistent MEA temperature and return the state.

        Parameters
        ----------
        cell : FuelCell
            Fully configured fuel-cell object.
        cell_conditions : CellConditions
            Operating conditions (same object passed to :meth:`set_initial_conditions`).
        initial_state : CellState
            State returned by :meth:`set_initial_conditions`.  Modified in place.

        Returns
        -------
        CellState
            The same object as *initial_state*, populated with all solved
            quantities including the self-consistent ``mea_temperature``.

        Raises
        ------
        RuntimeWarning
            Emitted (but not raised) if the root-finder did not converge.
        """
        state = initial_state
        state.thermal_resistance = self.thermal_model.heat_transfer_resistance(cell)
        x0 = self._initial_guess(cell, state)

        def residual(T_mea):
            self.thermal_model.set_mea_temperature(T_mea, cell, state)
            self.water_balance_model.calculate_water_transport(
                cell, state, gas_transport_model=self.gas_transport_model
            )
            self.gas_transport_model.calculate_gas_concentrations(cell, state)
            self.voltage_model.compute_cell_voltage(cell, state)
            T_mea_estimated = self.thermal_model.mea_temperature(
                cell, state, mea_temperature_estimation=True
            )
            return T_mea - T_mea_estimated

        result = root(residual, x0=x0, **self.root_kwargs)

        if not result.success:
            import warnings
            warnings.warn(
                f"ImplicitSteadyStateModel: root-finder did not converge — {result.message}",
                RuntimeWarning,
                stacklevel=2,
            )

        # Re-evaluate at the converged point so that `state` is fully consistent.
        residual(result.x)
        self._last_mea_temperature = result.x

        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_guess(self, cell, state) -> np.ndarray:
        """Return the initial guess for T_MEA.

        Uses the cached warm-start value when the shape matches; otherwise
        falls back to the explicit 0.7 V efficiency approximation.
        """
        T_explicit = self.thermal_model.mea_temperature(
            cell, state, mea_temperature_estimation=False
        )
        T_explicit = np.atleast_1d(T_explicit)

        if (
            self._last_mea_temperature is not None
            and np.atleast_1d(self._last_mea_temperature).shape == T_explicit.shape
        ):
            return self._last_mea_temperature

        return T_explicit
