"""
Cell model: implicit steady-state PEMFC performance model.

:class:`ImplicitSteadyStateModel` extends :class:`ExplicitSteadyStateModel`
by solving for the cell voltage and MEA temperature self-consistently.

The solve equation is:

    V = f(T_MEA(V))

where the MEA temperature is obtained from the heat balance:

    T_MEA = T_stack + R_th · (V_LHV − V) · i

and V is recomputed from the full physics (water balance, gas transport,
voltage model) at that T_MEA.  Because the problem is diagonal across
current-density points (each entry of V only affects the corresponding
entry of state), :func:`scipy.optimize.newton` (secant method) solves it
elementwise without building a dense Jacobian.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.optimize import newton

from .explicit_steady_state import ExplicitSteadyStateModel
from .state import CellState


@dataclass
class ImplicitSteadyStateModel(ExplicitSteadyStateModel):
    """
    Implicit steady-state model for PEMFC polarization curves.

    Unlike :class:`ExplicitSteadyStateModel`, which estimates T_MEA
    analytically, this model iterates until the heat released by the cell
    is consistent with the actual cell voltage.  The solve is fully
    vectorised: pass an array of current densities (and optionally
    array-valued condition fields) to evaluate all points in one call.

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
    """

    def solve(self, cell, cell_conditions, initial_state: CellState) -> CellState:
        """Solve for the self-consistent cell voltage and return the state.

        Parameters
        ----------
        cell : FuelCell
            Fully configured fuel-cell object.
        cell_conditions : CellConditions
            Operating conditions (same object passed to :meth:`set_initial_conditions`).
            All fields may be scalars or arrays of the same shape as
            ``current_density``.
        initial_state : CellState
            State returned by :meth:`set_initial_conditions`.  Modified in place.

        Returns
        -------
        CellState
            The same object as *initial_state*, populated with all solved
            quantities including the self-consistent ``mea_temperature``.
        """
        state = initial_state
        state.thermal_resistance = self.thermal_model.heat_transfer_resistance(cell)

        def _f(cell_voltage):
            mea_temperature = self.thermal_model.mea_temperature(cell, state, cell_voltage)
            self.thermal_model.set_mea_temperature(mea_temperature, cell, state)
            self.water_balance_model.calculate_water_transport(
                cell, state, gas_transport_model=self.gas_transport_model
            )
            self.gas_transport_model.calculate_gas_concentrations(cell, state)
            self.voltage_model.compute_cell_voltage(cell, state)
            if cell_voltage is None:
                return np.asarray(state.cell_voltage, dtype=float)
            return cell_voltage - state.cell_voltage

        # Warm start from the explicit 0.7 V estimate.
        cell_voltage_0 = _f(None)

        # Each entry of cell_voltage only affects the corresponding entry of
        # state (no cross-point coupling), so the Jacobian is diagonal — a
        # vectorised elementwise secant solve avoids a dense Jacobian.
        cell_voltage = newton(_f, x0=cell_voltage_0, disp=False, tol=1e-3, rtol=1e-3)
        _f(cell_voltage)
        return state
