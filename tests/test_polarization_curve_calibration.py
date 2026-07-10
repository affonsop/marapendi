"""Tests for SteadyStatePolarizationCurveCalibration and module-level helpers.

Uses a minimal synthetic dataset (2 cases, 3 current-density points each) so
tests run without disk I/O or expensive optimisations.
"""
import numpy as np
import pandas as pd
import pytest
import marapendi as mrpd
from marapendi.estimation.polarization_curve_calibration import (
    SteadyStatePolarizationCurveCalibration,
    optimal_n_1se,
    build_rmse_stats_df,
)
from marapendi.estimation.parameters import Parameter, UnknownParameter
from marapendi.models.base.explicit_steady_state import ExplicitSteadyStateModel


# ---------------------------------------------------------------------------
# Synthetic dataset fixtures
# ---------------------------------------------------------------------------

_CURRENT_DENSITIES = np.array([2e3, 5e3, 1e4])   # A/m²

_CONDITIONS = pd.DataFrame([
    dict(case=1, **{'cell-temperature': 353.15, 'pressure-ca': 1.5e5, 'pressure-an': 1.5e5,
                    'rh-ca': 0.50, 'rh-an': 0.50, 'st-ca': 2.0, 'st-an': 1.5}),
    dict(case=2, **{'cell-temperature': 323.15, 'pressure-ca': 2.5e5, 'pressure-an': 2.5e5,
                    'rh-ca': 0.30, 'rh-an': 0.30, 'st-ca': 2.0, 'st-an': 1.5}),
])

_EXP_ROWS = [
    {'case': c, 'current-density': i, 'voltage': 0.72, 'hfr': 5e-5}
    for c in [1, 2] for i in _CURRENT_DENSITIES
]
_EXPERIMENTAL = pd.DataFrame(_EXP_ROWS)


def _cell_creator(params):
    liq = mrpd.DarcyTransportModel(J_function_exponent=2)
    ionomer = mrpd.PFSAIonomer(
        equivalent_weight=params.get('memb-equiv-weight', 1100),
        reference_conductivity=50. * params.get('memb-cond-correction', 1.0),
    )
    return mrpd.FuelCell(
        area=25e-4,
        electric_resistance=params.get('elec-resistance', 30e-7),
        ca=mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(
                ecsa=60e3, platinum_loading=0.4e-2, ionomer=ionomer,
                reaction=mrpd.ElectrochemicalReaction(
                    reference_exchange_current_density=params.get('i0-c', 2.5e-4),
                    reaction_order=0.54, activation_energy=67e6,
                    reference_activity=1e5, reference_temperature=353.15,
                    number_of_electrons=2, charge_transfer_coeff=0.5,
                ),
                thickness=10e-6, thermal_conductivity=0.22,
                pore_diameter=40e-9, absolute_permeability=1e-13, contact_angle=97.,
                two_phase_transport_model=liq,
            ),
            gdl=mrpd.GasDiffusionLayer(
                thickness=200e-6, porosity=0.6, contact_angle=120.,
                tortuosity=2.0, absolute_permeability=1e-12,
                thermal_conductivity=0.5, two_phase_transport_model=liq,
            ),
            ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='o2'),
            thermal_contact_resistance=4e-4,
        ),
        an=mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(
                thickness=5e-6, two_phase_transport_model=liq,
            ),
            gdl=mrpd.GasDiffusionLayer(
                thickness=200e-6, tortuosity=2.0,
                thermal_conductivity=0.5, two_phase_transport_model=liq,
            ),
            ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='h2'),
            thermal_contact_resistance=4e-4,
        ),
        membrane=mrpd.PFSA(ionomer=ionomer, dry_thickness=25e-6),
    )


_KNOWN = [
    Parameter(value=1100.,  key='memb-equiv-weight'),
    Parameter(value=30e-7,  key='elec-resistance'),
]
_UNKNOWN = [
    UnknownParameter(value=2.5e-4, initial_guess=2.5e-4,
                     lower_bound=1e-5, upper_bound=1e-2, key='i0-c', is_linear=False),
    UnknownParameter(value=1.0, initial_guess=1.0,
                     lower_bound=0.1, upper_bound=20., key='memb-cond-correction', is_linear=True),
]


@pytest.fixture
def calibration():
    return SteadyStatePolarizationCurveCalibration(
        conditions_dataset=_CONDITIONS.copy(),
        experimental_dataset=_EXPERIMENTAL.copy(),
        cell_creator=_cell_creator,
        known_parameters=_KNOWN,
        unknown_parameters=_UNKNOWN,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_full_case_list(self, calibration):
        assert set(calibration.full_case_list) == {1, 2}

    def test_case_conditions_built(self, calibration):
        assert 1 in calibration.case_conditions
        assert 2 in calibration.case_conditions

    def test_case_conditions_are_cell_conditions(self, calibration):
        from marapendi.simulation.conditions import CellConditions
        assert isinstance(calibration.case_conditions[1], CellConditions)

    def test_hfr_mask_finite(self, calibration):
        assert np.all(calibration.hfr_mask[1])
        assert np.all(calibration.hfr_mask[2])

    def test_hfr_weight_factor_positive(self, calibration):
        assert calibration.hfr_weight_factor > 0

    def test_known_params_loaded(self, calibration):
        assert calibration.params['memb-equiv-weight'] == 1100.
        assert calibration.params['elec-resistance'] == 30e-7

    def test_unknown_params_at_initial_guess(self, calibration):
        assert calibration.params['i0-c'] == 2.5e-4
        assert calibration.params['memb-cond-correction'] == 1.0


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class TestDatasetHelpers:
    def test_get_case_dataset_filters_correctly(self, calibration):
        ds = calibration.get_case_dataset(1)
        assert (ds['case'] == 1).all()
        assert len(ds) == len(_CURRENT_DENSITIES)

    def test_make_conditions_current_density_shape(self, calibration):
        cond = calibration.case_conditions[1]
        n = len(_CURRENT_DENSITIES)
        assert np.atleast_1d(cond.current_density).shape == (n,)

    def test_make_conditions_temperature(self, calibration):
        cond = calibration.case_conditions[1]
        assert np.all(np.atleast_1d(cond.cell_temperature) == pytest.approx(353.15))

    def test_make_conditions_pressure_ca(self, calibration):
        cond = calibration.case_conditions[1]
        assert np.all(np.atleast_1d(cond.ca.outlet_pressure) == pytest.approx(1.5e5))


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

class TestSimulation:
    def test_build_cell_from_p_vector(self, calibration):
        p = calibration.p_initial_guess
        cell = calibration.build_cell_from_unknown_p_vector(p)
        assert isinstance(cell, mrpd.FuelCell)

    def test_simulate_voltage_plausible(self, calibration):
        p = calibration.p_initial_guess
        cell = calibration.build_cell_from_unknown_p_vector(p)
        V, hfr, state = calibration.simulate_voltage_and_hfr(cell, 1)
        assert np.all(np.atleast_1d(V) > 0.3)
        assert np.all(np.atleast_1d(V) < 1.23)

    def test_simulate_hfr_positive(self, calibration):
        p = calibration.p_initial_guess
        cell = calibration.build_cell_from_unknown_p_vector(p)
        _, hfr, _ = calibration.simulate_voltage_and_hfr(cell, 1)
        assert np.all(np.atleast_1d(hfr) > 0)

    def test_apply_hfr_weights_scales_up(self, calibration):
        raw = np.array([5e-5])
        weighted = calibration.apply_hfr_weights(raw)
        assert weighted[0] > raw[0]   # 1e4 × weight_factor > 1 always

    def test_build_y_exp_and_y_sim_same_shape(self, calibration):
        p = calibration.p_initial_guess
        y_exp = calibration.build_y_exp(1)
        y_sim = calibration.compute_y_sim(p, case_list=[1])
        assert y_exp.shape == y_sim.shape

    def test_compute_y_sim_over_all_cases(self, calibration):
        p = calibration.p_initial_guess
        y = calibration.compute_y_sim(p)
        assert y.ndim == 1
        # Each case contributes 2 × n_points (voltage + weighted HFR)
        assert len(y) == 2 * 2 * len(_CURRENT_DENSITIES)

    def test_compute_residuals_shape(self, calibration):
        p = calibration.p_initial_guess
        res = calibration.compute_residuals(p)
        y_exp = calibration.build_y_exp_cases(calibration.full_case_list)
        assert res.shape == y_exp.shape


# ---------------------------------------------------------------------------
# k-fold and parameter selection integration
# ---------------------------------------------------------------------------

class TestKFoldIntegration:
    def test_set_k_folds_with_two_cases(self, calibration):
        calibration.set_k_folds(k=2)
        assert len(calibration.k_folds) == 2
        all_cases = [c for fold in calibration.k_folds for c in fold]
        assert sorted(all_cases) == [1, 2]

    def test_subset_of_unknown_parameters(self, calibration):
        calibration.subset_of_unknown_parameters(keys=['i0-c'])
        assert calibration.n_unkown_p == 1
        assert calibration.p_i_name == ['i0-c']
        calibration.reset_unknown_parameters()

    def test_reset_restores_full_set(self, calibration):
        calibration.subset_of_unknown_parameters(keys=['i0-c'])
        calibration.reset_unknown_parameters()
        assert calibration.n_unkown_p == 2


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestOptimal1SE:
    def test_selects_simplest_within_1se(self):
        test_mean = pd.Series({5: 10.0, 10: 8.0, 15: 7.5, 20: 7.0})
        test_std  = pd.Series({5: 0.5,  10: 0.5, 15: 0.5,  20: 0.5})
        # best = n=20 (mean 7.0), threshold = 7.0 + 0.5 = 7.5
        # first model ≤ 7.5 is n=15
        assert optimal_n_1se(test_mean, test_std) == 15

    def test_returns_best_when_all_within_1se(self):
        test_mean = pd.Series({5: 8.0, 10: 7.0})
        test_std  = pd.Series({5: 2.0, 10: 2.0})
        # threshold = 7.0 + 2.0 = 9.0; first n with mean ≤ 9.0 is n=5
        assert optimal_n_1se(test_mean, test_std) == 5


class TestBuildRmseStatsDf:
    def _make_rmse_df(self):
        rows = []
        for n in [5, 10]:
            for fold_id in [0, 1]:
                for case in [1, 2]:
                    rows.append(dict(rmse=0.02, case=case, fold_id=fold_id,
                                     n_params=n, is_test=(fold_id == 0)))
        return pd.DataFrame(rows)

    def test_returns_dataframe(self):
        df = build_rmse_stats_df(self._make_rmse_df())
        assert isinstance(df, pd.DataFrame)

    def test_index_is_n_params(self):
        df = build_rmse_stats_df(self._make_rmse_df())
        assert set(df.index) == {5, 10}

    def test_summary_columns_present(self):
        df = build_rmse_stats_df(self._make_rmse_df())
        assert 'Mean' in df.columns
        assert 'Median' in df.columns
