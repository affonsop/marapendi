"""
Parameter estimation glue between :mod:`marapendi` cell models and
experimental datasets.

Classes
-------
Case
    One experimental case: a :class:`~marapendi.cell.Cell`, its
    :class:`~marapendi.conditions.CellOperatingConditions`, and a
    dict of experimental data arrays.
ParameterEstimation
    Simulates :class:`Case` objects with a
    :class:`~marapendi.model.CellModel` and assembles simulated /
    experimental datasets for fitting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..cell import Cell
from ..conditions import CellOperatingConditions
from ..model import CellModel
from ..state import CellState


@dataclass
class Case:
    """One experimental case: a cell, its operating conditions, and experimental data.

    Attributes
    ----------
    cell : Cell
        The cell configuration (static physical properties) for this case.
    conditions : CellOperatingConditions
        Operating conditions at which to evaluate the cell model.
    exp_data : dict[str, np.ndarray]
        Experimental data, keyed by output-quantity name (e.g.
        ``'cell_voltage'``, ``'hfr'``). Each value is an array matching the
        shape of ``conditions.current_density``. Keys must be registered in
        :attr:`ParameterEstimation.OUTPUT_EXTRACTORS`.
    name : str
        Optional human-readable identifier for the case.
    """

    cell: Cell
    conditions: CellOperatingConditions
    exp_data: dict[str, np.ndarray] = field(default_factory=dict)
    name: str = ''


@dataclass
class ParameterEstimation:
    """Simulates :class:`Case` objects with a :class:`CellModel`.

    Attributes
    ----------
    cell_model : CellModel
        Orchestrator used to compute the steady-state solution. Its ``cell``
        attribute is overwritten by :meth:`simulate_model` for each case.
    extract_case_data : Callable[[dict[str, np.ndarray], CellState], tuple[np.ndarray, np.ndarray]]
        User-supplied function ``(exp_data, state) -> (y_sim, y_exp)``. Given
        a case's ``exp_data`` and the simulated :class:`CellState`, it must
        return the simulated and experimental output vectors for that case,
        in matching order/shape. This is where the mapping from raw state
        quantities (e.g. ``state.cell_voltage``) to experimental data keys
        is defined, so :class:`ParameterEstimation` stays agnostic to what
        is being fitted.
    cases : list[Case]
        Experimental cases to simulate/compare.
    """

    cell_model: CellModel
    extract_case_data: Callable[[dict[str, np.ndarray], CellState], tuple[np.ndarray, np.ndarray]]
    cases: list[Case] = field(default_factory=list)

    def simulate_model(self, conditions: CellOperatingConditions, cell: Cell) -> CellState:
        """Compute the steady-state solution of ``cell`` under ``conditions``."""
        self.cell_model.cell = cell
        return self.cell_model.steady_state_solution(conditions)

    def build_dataset(self, cases: list[Case] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate ``cases`` and stack inputs/outputs for fitting.

        Parameters
        ----------
        cases : list[Case], optional
            Cases to simulate (defaults to :attr:`cases`).

        Returns
        -------
        x : np.ndarray
            Concatenated current densities, tiled to the size of
            ``y_sim``/``y_exp`` for each case (e.g. if
            :attr:`extract_case_data` concatenates several quantities per
            case).
        y_sim : np.ndarray
            Concatenated simulated outputs, as returned by
            :attr:`extract_case_data` for each case.
        y_exp : np.ndarray
            Concatenated experimental outputs, matching ``y_sim``.
        """
        cases = cases if cases is not None else self.cases
        x_parts, y_sim_parts, y_exp_parts = [], [], []
        for case in cases:
            state = self.simulate_model(case.conditions, case.cell)
            y_sim, y_exp = self.extract_case_data(case.exp_data, state)
            y_sim, y_exp = np.atleast_1d(y_sim), np.atleast_1d(y_exp)
            x = np.atleast_1d(case.conditions.current_density)
            # extract_case_data may concatenate several quantities (e.g. voltage
            # and HFR) per case, so tile x to match if it divides evenly.
            x = np.tile(x, y_sim.size // x.size) if y_sim.size % x.size == 0 else np.broadcast_to(x, y_sim.shape)
            x_parts.append(x)
            y_sim_parts.append(y_sim)
            y_exp_parts.append(y_exp)
        return np.concatenate(x_parts), np.concatenate(y_sim_parts), np.concatenate(y_exp_parts)

    def compare_sim_exp(self, case: Case) -> dict[str, np.ndarray]:
        """Simulate a single ``case`` and return its simulated vs. experimental data.

        Returns
        -------
        dict
            ``{'current_density': ..., 'sim': ..., 'exp': ...}``.
        """
        state = self.simulate_model(case.conditions, case.cell)
        y_sim, y_exp = self.extract_case_data(case.exp_data, state)
        return {
            'current_density': np.atleast_1d(case.conditions.current_density),
            'sim': np.atleast_1d(y_sim),
            'exp': np.atleast_1d(y_exp),
        }
