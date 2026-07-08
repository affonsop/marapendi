"""
Steady-state polarisation curve calibration against voltage and HFR data.

:class:`SteadyStatePolarizationCurveCalibration` extends
:class:`~marapendi.estimation.BaseModelCalibration` to calibrate PEMFC performance
models.  It builds the simulated-vs-experimental residual vector from voltage and
high-frequency resistance (HFR) measurements across multiple operating conditions,
and provides helpers to run and post-process k-fold cross-validation.

Module-level helpers (:func:`collect_rmse_df`, :func:`optimal_n_1se`,
:func:`build_rmse_stats_df`, :func:`rmse_complexity_latex`,
:func:`rmse_complexity_table`) are used by the plotting functions in
:mod:`marapendi.estimation.plots`.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..simulation.conditions import CellConditions, SideConditions
from ..models.base.explicit_steady_state import ExplicitSteadyStateModel
from .base_calibration import BaseModelCalibration


@dataclass
class SteadyStatePolarizationCurveCalibration(BaseModelCalibration):
    """Calibration of a steady-state polarisation curve model against voltage and HFR data.

    conditions_dataset : per-case operating conditions (temperature, pressures, RH, stoichiometries).
    experimental_dataset : per-case measurements with columns case, current-density, voltage, hfr.
    cell_creator : callable(params) → FuelCell used to build the model from a parameter dict.
    cell_model : solver instance; defaults to ExplicitSteadyStateModel.
    """
    conditions_dataset: pd.DataFrame
    experimental_dataset: pd.DataFrame
    cell_creator: Callable
    cell_model: ExplicitSteadyStateModel = field(default_factory=ExplicitSteadyStateModel)

    def __post_init__(self):
        BaseModelCalibration.__post_init__(self)
        self.full_case_list = self.experimental_dataset['case'].unique()

        self.populate_exp_dataset_conditions()
        self.build_cases_conditions()

        self.hfr_mask = {case: np.isfinite(self.get_case_dataset(case)['hfr']) for case in self.full_case_list}
        self.hfr_weight_factor = (
            np.sum(np.isfinite(self.experimental_dataset['voltage']))
            / np.sum(np.isfinite(self.experimental_dataset['hfr']))
        )

    def solve_case(self, cell, case):
        """Run the steady-state solver for *case* and return the final :class:`CellState`."""
        cond = self.case_conditions[case]
        state = self.cell_model.set_initial_conditions(cell, cond)
        return self.cell_model.solve(cell, cond, state)

    def apply_hfr_weights(self, hfr):
        """Scale HFR values so their residuals are comparable in magnitude to voltage residuals.

        Multiplies by 1e4 (unit conversion from Ω·m² to mΩ·cm²) and by
        ``hfr_weight_factor`` (ratio of voltage to HFR measurement counts).
        """
        return hfr * 1e4 * self.hfr_weight_factor

    def build_cell_from_unknown_p_vector(self, unknown_p_vector):
        """Build a :class:`FuelCell` from the current known params merged with *unknown_p_vector*."""
        px = self.params.copy()
        px.update(dict(zip(self.p_i_name, unknown_p_vector)))
        cell = self.cell_creator(px)
        return cell

    def build_y_sim_cases(self, cell, case_list):
        """Concatenate simulated output vectors for all cases in *case_list*."""
        return np.concatenate([
            self.build_y_sim(cell, case) for case in case_list
        ])

    def simulate_voltage_and_hfr(self, cell, case):
        """Solve *case* and return ``(cell_voltage, hfr, state)``."""
        state = self.solve_case(cell, case)
        hfr = self.cell_model.voltage_model.high_frequency_resistance(cell, state)
        return state.cell_voltage, hfr, state

    def build_y_sim(self, cell, case):
        """Return the concatenated [voltage | weighted HFR] simulated vector for *case*."""
        cell_voltage, hfr, _ = self.simulate_voltage_and_hfr(cell, case)
        hfr_sim = self.apply_hfr_weights(hfr) * self.hfr_mask[case]
        return np.concatenate([cell_voltage, hfr_sim])

    def build_y_exp_cases(self, case_list):
        """Concatenate experimental output vectors for all cases in *case_list*."""
        return np.concatenate([
            self.build_y_exp(case) for case in case_list
        ])

    def build_y_exp(self, case):
        """Return the concatenated [voltage | weighted HFR] experimental vector for *case*."""
        case_dataset = self.get_case_dataset(case)
        hfr_exp = self.apply_hfr_weights(case_dataset['hfr']) * self.hfr_mask[case]
        return np.concatenate([case_dataset['voltage'], hfr_exp])

    def compute_y_sim(self, unknown_p_vector=None, cell=None, case_list=[]):
        """Return the concatenated simulated output vector for all cases.

        Either *unknown_p_vector* (physical values) or *cell* must be provided.
        When *case_list* is empty, all cases in ``self.full_case_list`` are used.
        """
        if len(case_list) == 0:
            case_list = self.full_case_list
        if (unknown_p_vector is not None) and (cell is None):
            cell = self.build_cell_from_unknown_p_vector(unknown_p_vector)
        return self.build_y_sim_cases(cell, case_list)

    def compute_residuals(self, unknown_p_vector=None, cell=None, case_list=[]):
        """Return element-wise residuals (y_exp − y_sim) for *case_list*."""
        if len(case_list) == 0:
            case_list = self.full_case_list
        return self.build_y_exp_cases(case_list) - self.compute_y_sim(unknown_p_vector, cell, case_list)

    def populate_exp_dataset_conditions(self):
        """Merge per-case conditions into experimental_dataset and shift current-density by +1 A/m² to avoid division-by-zero at OCV."""
        for side in ('ca', 'an'):
            if f'min-current-at-st-{side}' not in self.conditions_dataset.columns:
                self.conditions_dataset[f'min-current-at-st-{side}'] = 0
        self.experimental_dataset['current-density'] += 1

        for column in self.conditions_dataset.columns:
            if column not in self.experimental_dataset.columns:
                self.experimental_dataset = self.experimental_dataset.merge(
                    self.conditions_dataset[['case', column]], on='case'
                )

    def get_case_dataset(self, case):
        """Return the subset of ``experimental_dataset`` for *case*."""
        return self.experimental_dataset[self.experimental_dataset['case'] == case]

    def build_cases_conditions(self, current_density=None):
        """Build and store a :class:`CellConditions` object for every case in ``full_case_list``."""
        self.case_conditions = {}
        for case in self.full_case_list:
            self.case_conditions[case] = self.make_conditions(case, current_density)

    def make_conditions(self, case, current_density=None):
        """Build CellConditions for a case. If current_density is given, all tabulated columns are interpolated onto it."""
        case_dataset = self.get_case_dataset(case)
        xp = case_dataset['current-density'].values

        def interp(col):
            return np.interp(current_density, xp, case_dataset[col].values)

        if current_density is None:
            current_density = xp
            interp = lambda col: case_dataset[col].values  # noqa: E731

        i = current_density
        stoich_ca = interp('st-ca') * np.maximum(interp('min-current-at-st-ca') / i, 1)
        stoich_an = interp('st-an') * np.maximum(interp('min-current-at-st-an') / i, 1)

        return CellConditions(
            current_density=i,
            cell_temperature=interp('cell-temperature'),
            ca=SideConditions(
                inlet_temperature=interp('cell-temperature'),
                outlet_pressure=interp('pressure-ca'),
                inlet_relative_humidity=interp('rh-ca'),
                dry_o2_mole_fraction=0.21,
                stoichiometry=stoich_ca,
            ),
            an=SideConditions(
                inlet_temperature=interp('cell-temperature'),
                outlet_pressure=interp('pressure-an'),
                inlet_relative_humidity=interp('rh-an'),
                dry_h2_mole_fraction=1.0,
                stoichiometry=stoich_an,
            ),
        )

    def get_estimated_parameters_from_fold_results(self, fold_results):
        """Extract the estimated-parameter columns from a fold-results DataFrame row."""
        return fold_results.loc[:, fold_results.columns[4:]].copy()

    def simulate_for_fold_results(self, fold_results):
        """Build a cell from *fold_results* and simulate all cases; return voltage, HFR, and state dicts."""
        estimated_parameters = self.get_estimated_parameters_from_fold_results(fold_results)
        estimated_parameters.dropna(axis=1, inplace=True)
        estimated_dict = dict(
            zip(
                estimated_parameters.columns.values,
                estimated_parameters.values[0],
            )
        )
        px = self.params.copy()
        px.update(estimated_dict)
        cell = self.cell_creator(px)

        voltage_cases, hfr_cases, state_cases = {}, {}, {}
        for case in self.full_case_list:
            voltage, hfr, state = self.simulate_voltage_and_hfr(cell, case)
            voltage_cases[case] = voltage
            hfr_cases[case] = hfr
            state_cases[case] = state
        return voltage_cases, hfr_cases, state_cases


def collect_rmse_df(model, cv_results, variable, quantity_multiplier):
    """Compute per-case, per-fold RMSE from cross-validation results and return a tidy DataFrame.

    Parameters
    ----------
    model : SteadyStatePolarizationCurveCalibration
    cv_results : pd.DataFrame
        Output of :meth:`run_k_fold_cross_validation`.
    variable : str
        ``'voltage'`` or ``'hfr'``.
    quantity_multiplier : float
        Unit conversion factor applied to residuals before computing RMSE.
    """
    rmse_values, case_column, fold_id_column, complexity_column, is_test_column = [], [], [], [], []

    for n_params in cv_results.n_params.unique():
        folds_results = cv_results[cv_results.n_params == n_params]

        for fold_id in cv_results.fold_id.unique():
            fold_id = int(fold_id)
            fold_results = folds_results[folds_results.fold_id == fold_id]
            voltage_cases, hfr_cases, _ = model.simulate_for_fold_results(fold_results)

            for case in model.full_case_list:
                y_sim = voltage_cases[case] if variable == 'voltage' else hfr_cases[case] * model.hfr_mask[case]
                case_dataset = model.get_case_dataset(case)
                y_exp = np.nan_to_num(case_dataset[variable])

                residuals = y_exp - y_sim
                n_valid = sum(1 - np.isnan(case_dataset[variable]))
                rmse = np.sqrt(np.dot(residuals, residuals) / n_valid) * quantity_multiplier

                rmse_values.append(rmse)
                case_column.append(case)
                fold_id_column.append(fold_id)
                complexity_column.append(n_params)
                is_test_column.append(case in model.k_folds[fold_id])

    return pd.DataFrame({
        "rmse": rmse_values,
        "case": case_column,
        "fold_id": fold_id_column,
        "n_params": complexity_column,
        "is_test": is_test_column,
    })


def optimal_n_1se(test_mean, test_std):
    """Return the simplest complexity whose mean test RMSE is within 1 SE of the minimum (1-SE rule).

    Reference: Hastie, T. et al. "The Elements of Statistical Learning", §7.3.
    """
    # 1-SE rule: simplest model within 1 std of the best
    # See https://esl.hohoweiya.xyz/book/The%20Elements%20of%20Statistical%20Learning.pdf
    best_n = test_mean.idxmin()
    threshold = test_mean[best_n] + test_std[best_n]
    return next(n for n in test_mean.index if test_mean[n] <= threshold)


def build_rmse_stats_df(rmse_vs_complexity_df):
    """Aggregate per-case RMSE values into summary statistics by complexity level.

    Returns a DataFrame indexed by ``n_params`` with one column per fold and
    aggregate columns (Min., Max., Median, Mean).
    """
    test_rows = (
        rmse_vs_complexity_df[rmse_vs_complexity_df["is_test"]]
        .groupby(['n_params', 'fold_id']).mean()
        .reset_index()
    )

    per_case = test_rows.pivot(index='n_params', columns='fold_id', values='rmse')

    stats_df = (
        test_rows.groupby('n_params')
        .agg(['min', 'max', 'median', 'mean'])
        .drop(columns=['case', 'fold_id'], level=0)
    )
    stats_df.columns = ['_'.join(col) for col in stats_df.columns]
    stats_df = stats_df.join(per_case)

    case_cols = [c for c in stats_df.columns if not (isinstance(c, str) and '_' in c)]
    stat_cols = [c for c in stats_df.columns if c not in case_cols]

    stats_df = stats_df[case_cols + stat_cols]
    stats_df = stats_df.rename(columns={
        c: f'{c:.0f}' for c in case_cols
    } | {
        'rmse_min': 'Min.',
        'rmse_max': 'Max.',
        'rmse_median': 'Median',
        'rmse_mean': 'Mean',
    })

    return stats_df


def rmse_complexity_latex(stats_df):
    """Render *stats_df* as a LaTeX booktabs table string."""
    n_cases = sum(1 for c in stats_df.columns if c not in ('Min.', 'Max.', 'Median', 'Mean'))
    return (
        stats_df.to_latex(
            float_format="%.1f",
            na_rep="-",
            caption="RMSE vs complexity",
            label="tab:rmse_complexity",
            position="h",
        ).replace(
            r'\toprule',
            r'\toprule' + '\n' +
            rf' & \multicolumn{{{n_cases}}}{{c}}{{Cases}} \\' + '\n' +
            rf'\cmidrule(lr){{2-{n_cases + 1}}}',
        )
    )


def rmse_complexity_table(rmse_vs_complexity_df, filename=None):
    """Build the RMSE summary table and its LaTeX representation.

    Optionally writes the LaTeX string to *filename*.  Returns ``(stats_df, latex)``.
    """
    stats_df = build_rmse_stats_df(rmse_vs_complexity_df)
    latex = rmse_complexity_latex(stats_df)

    if filename:
        with open(filename, 'w') as f:
            f.write(latex)

    return stats_df, latex
