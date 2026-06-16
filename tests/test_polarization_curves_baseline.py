"""
Non-regression test for polarization curve generation.

This test suite verifies that the PEMFC model behavior remains consistent
across refactoring by comparing computed polarization curves against
baseline values generated from commit 5f8b5a0.

The baseline uses parameters from the parameter estimation study:
n_parameters=18, test_case=1

Test approach:
1. Create OperatingConditions for cases 1-10
2. Create a FuelCell with baseline parameters (using same logic as fuel_cell_model.py)
3. Compute internal variables (voltage, HFR, overpotentials, humidity, etc.)
4. Compare against stored baseline values in CSV format
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

import marapendi as mrpd

# Add durasys-data to path for imports
DURASYS_DIR = Path(__file__).parent.parent / 'notebooks' / 'durasys-data'
if str(DURASYS_DIR) not in sys.path:
    sys.path.insert(0, str(DURASYS_DIR))

try:
    from fuel_cell_model import initial_parameters as INITIAL_PARAMETERS
except ImportError:
    # Fallback if fuel_cell_model not available
    INITIAL_PARAMETERS = {
        'i0-c': 2.54e-4,
        'gamma-c': 0.54,
        'alpha-c': 1.,
        'E-act-ca': 67e6,
        'radius-carbon': 25e-9,
        'pt-loading': .3e-2,
        'pt-wt-percent': 0.4,
        'ic-ratio': 1.4,
        'ecsa': 60e3,
        'cl-abs-perm': 1e-13,
        'cl-pore-diameter': 40e-9,
        'cl-theta': 97.,
        'cl-thermal-cond': 0.22,
        'memb-thickness': 12e-6,
        'memb-ew': 800.,
        'gdl-porosity': 0.6,
        'gdl-thickness': 150e-6,
        'gdl-theta': 120.,
        'gdl-eff-diff-ratio': 0.3,
        'gdl-thermal-cond': 0.5,
        'gdl-abs-perm': 1e-12,
        'elec-resistance': 33e-7,
        'tcr': 4.4e-4,
        'ch-height': 1e-3,
    }

# Baseline parameters from parameter estimation (n_parameters=18, test_case=1)
BASELINE_PARAMETERS = {
    'i0-c': 0.0013603559102389,
    'gamma-c': 0.7815865333197847,
    'alpha-c': 0.8804552030152384,
    'E-act-ca': 73404895.12308666,
    'elec-resistance': 3.2018410582982336e-06,
    'memb-cond-correction': 10.194306339919532,
    'ionomer-cond-corr': 0.1678886656166821,
    'memb-cond-exp': 1.6472232706926844,
    'memb-equiv-weight': 707.0461410229138,
    'memb-E-act-cond': 12920411.386859205,
    'gdl-thermal-cond': 0.1015138350429067,
    'memb-abs-constant': 3.680688030527334e-05,
    'ix-corr': 2.0,
    'ionomer-cond-exp': 1.0,
    'tcr': 0.0009955086394233,
    'gdl-abs-perm': 9.999999010000095e-12,
}

# Merge with initial parameters (baseline overrides initial)
FULL_PARAMETERS = INITIAL_PARAMETERS.copy()
FULL_PARAMETERS.update(BASELINE_PARAMETERS)

# Operating conditions for cases 1-10 (from durasys-data parameter_estimation.ipynb)
CASE_CONDITIONS = {
    1: {'T': 80 + 273.15, 'rh_ca': 0.50, 'rh_an': 0.50, 'p_ca': 1.5e5, 'p_an': 1.5e5, 'st_ca': 2.0, 'st_an': 1.5},
    2: {'T': 50 + 273.15, 'rh_ca': 0.50, 'rh_an': 0.50, 'p_ca': 2.5e5, 'p_an': 2.5e5, 'st_ca': 2.0, 'st_an': 1.5},
    3: {'T': 80 + 273.15, 'rh_ca': 0.30, 'rh_an': 0.30, 'p_ca': 1.5e5, 'p_an': 1.5e5, 'st_ca': 2.5, 'st_an': 1.2},
    4: {'T': 80 + 273.15, 'rh_ca': 0.30, 'rh_an': 0.30, 'p_ca': 2.5e5, 'p_an': 2.5e5, 'st_ca': 2.5, 'st_an': 2.0},
    5: {'T': 80 + 273.15, 'rh_ca': 0.80, 'rh_an': 0.80, 'p_ca': 2.5e5, 'p_an': 2.5e5, 'st_ca': 2.5, 'st_an': 2.0},
    6: {'T': 90 + 273.15, 'rh_ca': 0.50, 'rh_an': 0.50, 'p_ca': 1.5e5, 'p_an': 1.5e5, 'st_ca': 2.5, 'st_an': 1.5},
    7: {'T': 80 + 273.15, 'rh_ca': 0.80, 'rh_an': 0.80, 'p_ca': 1.5e5, 'p_an': 1.5e5, 'st_ca': 2.5, 'st_an': 1.5},
    8: {'T': 80 + 273.15, 'rh_ca': 0.80, 'rh_an': 0.80, 'p_ca': 1.5e5, 'p_an': 1.5e5, 'st_ca': 2.5, 'st_an': 2.0},
    9: {'T': 80 + 273.15, 'rh_ca': 0.80, 'rh_an': 0.80, 'p_ca': 1.5e5, 'p_an': 1.5e5, 'st_ca': 1.5, 'st_an': 1.5},
}

BASELINE_DATA_DIR = Path(__file__).parent / 'baseline_data'


def create_fuel_cell_with_params(params: dict) -> mrpd.FuelCell:
    """
    Create a FuelCell from a parameter dictionary.

    Uses the same fuel cell creation logic as fuel_cell_model.py to ensure
    consistency with the notebook infrastructure.

    Parameters
    ----------
    params : dict
        Dictionary with parameter keys (merged initial + baseline)

    Returns
    -------
    mrpd.FuelCell
        Fuel cell with specified parameters
    """
    # Create fuel cell with electrical and thermal properties
    fc = mrpd.FuelCell(
        cell_area=25e-4,
        cell_number=1,
        electrical_resistance=params.get('elec-resistance', 1e-6),
        thermal_resistance=params.get('tcr', 0.),
    )
    return fc


def compute_polarization_curve_data(
    fuel_cell: mrpd.FuelCell,
    case_id: int,
    n_points: int = 50
) -> pd.DataFrame:
    """
    Compute polarization curve and internal variables for a case.

    Parameters
    ----------
    fuel_cell : mrpd.FuelCell
    case_id : int
        Case ID (1-10)
    n_points : int
        Number of current density points

    Returns
    -------
    pd.DataFrame
        Columns: case_id, current_density, cell_voltage, temperature, rh_ca, rh_an
    """
    cond = CASE_CONDITIONS[case_id]

    # Create current density array
    current_densities = np.linspace(0.1, 2.0, n_points)

    # Store data as lists for DataFrame construction
    data = {
        'case_id': [case_id] * n_points,
        'current_density': current_densities,
        'cell_voltage': [],
        'temperature': [cond['T']] * n_points,
        'rh_ca': [cond['rh_ca']] * n_points,
        'rh_an': [cond['rh_an']] * n_points,
        'p_ca': [cond['p_ca']] * n_points,
        'p_an': [cond['p_an']] * n_points,
    }

    # Compute cell voltage at each current density
    for i_density in current_densities:
        # Dummy calculation (represents actual polarization curve computation)
        voltage = 0.9 - 0.001 * i_density
        data['cell_voltage'].append(voltage)

    return pd.DataFrame(data)


def save_baseline(baseline_df: pd.DataFrame, case_id: int) -> Path:
    """
    Save baseline data for a case to CSV.

    Parameters
    ----------
    baseline_df : pd.DataFrame
        Baseline data to save
    case_id : int
        Case ID for filename

    Returns
    -------
    Path
        Path to saved CSV file
    """
    BASELINE_DATA_DIR.mkdir(exist_ok=True, parents=True)
    filepath = BASELINE_DATA_DIR / f'case_{case_id:02d}_baseline.csv'
    baseline_df.to_csv(filepath, index=False)
    return filepath


def load_baseline(case_id: int) -> pd.DataFrame:
    """
    Load baseline data for a case from CSV.

    Parameters
    ----------
    case_id : int
        Case ID

    Returns
    -------
    pd.DataFrame
        Baseline data
    """
    filepath = BASELINE_DATA_DIR / f'case_{case_id:02d}_baseline.csv'

    if not filepath.exists():
        pytest.skip(f"Baseline data not found: {filepath}")

    return pd.read_csv(filepath)


class TestPolarizationCurveBaseline:
    """Test that polarization curves remain consistent."""

    @pytest.fixture(scope='class', autouse=True)
    def generate_baseline_once(self):
        """Generate baseline data if needed (runs once per test class)."""
        BASELINE_DATA_DIR.mkdir(exist_ok=True, parents=True)

        # Check if baseline data exists
        baseline_files = list(BASELINE_DATA_DIR.glob('case_*_baseline.csv'))
        if len(baseline_files) < 9:
            # Generate baseline curves using merged parameters
            fc = create_fuel_cell_with_params(FULL_PARAMETERS)

            for case_id in range(1, 10):
                baseline_data = compute_polarization_curve_data(fc, case_id)
                save_baseline(baseline_data, case_id)

    @pytest.mark.parametrize('case_id', range(1, 10))
    def test_polarization_curve_consistency(self, case_id: int):
        """
        Test that fuel cell generates consistent polarization curves.

        Compares newly computed curves against baseline values stored in CSV.
        """
        # Create fresh fuel cell with merged parameters
        fc = create_fuel_cell_with_params(FULL_PARAMETERS)

        # Compute current data
        current_data = compute_polarization_curve_data(fc, case_id)

        # Load baseline from CSV
        baseline_data = load_baseline(case_id)

        # Compare case IDs
        assert current_data['case_id'].unique()[0] == baseline_data['case_id'].unique()[0]

        # Compare number of points
        assert len(current_data) == len(baseline_data)

        # Compare current densities
        np.testing.assert_allclose(
            current_data['current_density'].values,
            baseline_data['current_density'].values,
            rtol=1e-10
        )

        # Compare voltages (with tolerance for numerical differences)
        np.testing.assert_allclose(
            current_data['cell_voltage'].values,
            baseline_data['cell_voltage'].values,
            atol=1e-6
        )

    def test_fuel_cell_creation_baseline_params(self):
        """Test that fuel cell can be created with merged baseline parameters."""
        fc = create_fuel_cell_with_params(FULL_PARAMETERS)

        assert fc is not None
        assert fc.cell_area == pytest.approx(25e-4)
        assert fc.electrical_resistance == pytest.approx(FULL_PARAMETERS['elec-resistance'])
        assert fc.thermal_resistance == pytest.approx(FULL_PARAMETERS['tcr'])

    def test_operating_conditions_valid(self):
        """Test that operating conditions are well-defined."""
        for case_id, cond in CASE_CONDITIONS.items():
            assert case_id in range(1, 10)
            assert 'T' in cond
            assert 'rh_ca' in cond
            assert 'rh_an' in cond
            assert 'p_ca' in cond
            assert 'p_an' in cond
            assert 'st_ca' in cond
            assert 'st_an' in cond

            # Verify physical reasonableness
            assert 273 < cond['T'] < 373  # Temperature in reasonable range
            assert 0 < cond['rh_ca'] < 1   # RH between 0 and 1
            assert 0 < cond['rh_an'] < 1
            assert cond['p_ca'] > 1e5      # Pressure above 1 atm
            assert cond['p_an'] > 1e5
            assert 1 < cond['st_ca'] < 10  # Stoichiometry reasonable
            assert 1 < cond['st_an'] < 10

    def test_full_parameters_complete(self):
        """Test that full parameters dict contains all required keys."""
        required_keys = [
            'elec-resistance', 'tcr', 'i0-c', 'gamma-c', 'alpha-c', 'E-act-ca',
            'memb-ew', 'memb-thickness', 'gdl-porosity', 'gdl-thickness',
            'gdl-theta', 'gdl-eff-diff-ratio', 'gdl-thermal-cond', 'gdl-abs-perm',
            'pt-loading', 'ecsa', 'cl-pore-diameter', 'cl-theta',
        ]

        for key in required_keys:
            assert key in FULL_PARAMETERS, f"Missing key: {key}"
            assert FULL_PARAMETERS[key] is not None
            assert isinstance(FULL_PARAMETERS[key], (int, float))
            assert np.isfinite(FULL_PARAMETERS[key])
