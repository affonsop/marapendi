"""
Non-regression test for polarization curve generation.

Reference model  : notebooks/durasys-data/fuel_cell_model_legacy.py
                   (physics identical; import adapted from marapendi.legacy →
                    marapendi, which is the installed package at this commit)
Baseline params  : n_parameters=18, test_case=1 from
                   results_final_estimation_model2_new_perm_lim_cv.csv

All model construction and operating-condition code is copied here so that
the test has no dependency on the notebooks/ directory.

Workflow
--------
First run (no baseline CSVs present):
  pytest tests/test_polarization_curves_baseline.py  →  generates & saves

Subsequent runs:
  pytest tests/test_polarization_curves_baseline.py  →  compares against saved
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import marapendi as mrpd


# ---------------------------------------------------------------------------
# Parameters  (verbatim from fuel_cell_model_legacy.py → initial_parameters)
# ---------------------------------------------------------------------------

INITIAL_PARAMETERS: dict = {
    'i0-c': 2.54e-4,
    'gamma-c': 0.54,
    'alpha-c': 1.,
    'radius-carbon': 25e-9,
    'ionomer-E-act-cond': 15e6,
    'n_s': 2,
    'memb-E-act-cond': 15e6,
    'ionomer-cond-corr': 1,
    'ionomer-cond-exp': 1.5,
    'E-act-ca': 67e6,
    'ionomer-k1': 8.5,
    'ionomer-k2': 5.4,
    'ionomer-k3': 5.4,
    'alpha-w': 1,
    'gdl-porosity': 0.6,
    'elec-resistance': 33e-7,
    'tcr': 4.4e-4,
    'pt-wt-percent': 0.4,
    'Sh': 3.6, 'B_ch': 1., 'ch-height': 1e-3,
    'gdl-thickness': 150e-6,
    'gdl-theta': 120.,
    'gdl-eff-diff-ratio': 0.3,
    'gdl-thermal-cond': 0.5,
    'gdl-abs-perm': 1e-12,
    'cl-abs-perm': 1e-13,
    'ix-corr': 1.,
    'wet-transition': 0.4,
    'pt-loading': .3e-2,
    'ic-ratio': 1.4,
    'ecsa': 60e3,
    'memb-thickness': 12e-6,
    'memb-equiv-weight': 1100.,
    'memb-cond-correction': 1.0,
    'memb-water-diff': 2e-10,
    'memb-abs-constant': 1e-5,
    'memb-cond-exp': 1.5,
    'E-act-memb-diff': 20e6,
    'E-act-memb-abs': 20e6,
    'cl-theta': 97.,
    'cl-thermal-cond': 0.22,
    'cl-pore-diameter': 40e-9,
    'eod-parallel': False,
    'sorption-driving-force': False,
}

# Estimated parameters  (n_parameters=18, test_case=1)
_ESTIMATED: dict = {
    'i0-c':                 0.0013603559102389,
    'gamma-c':              0.7815865333197847,
    'alpha-c':              0.8804552030152384,
    'E-act-ca':             73404895.12308666,
    'elec-resistance':      3.2018410582982336e-06,
    'memb-cond-correction': 10.194306339919532,
    'B_ch':                 1.3173241932454605,
    'ionomer-cond-corr':    0.1678886656166821,
    'memb-cond-exp':        1.6472232706926844,
    'Sh':                   0.7956740630180096,
    'memb-equiv-weight':    707.0461410229138,
    'memb-E-act-cond':      12920411.386859205,
    'gdl-thermal-cond':     0.1015138350429067,
    'memb-abs-constant':    3.680688030527334e-05,
    'ix-corr':              2.0,
    'ionomer-cond-exp':     1.0,
    'tcr':                  0.0009955086394233,
    'gdl-abs-perm':         9.999999010000095e-12,
}

# Merged: initial values overridden by estimated ones
FULL_PARAMETERS: dict = {**INITIAL_PARAMETERS, **_ESTIMATED}


# ---------------------------------------------------------------------------
# Operating conditions
# (variations = [0, 1, 2, 4, 15, 6, 5, 8, 10] → cases 1-9)
# (conditions table from treat_data.py, indexed by Variation)
#
# Fields: T [K], rh_ca [-], rh_an [-], p_ca [Pa], p_an [Pa], st_ca, st_an
# ---------------------------------------------------------------------------

CASE_CONDITIONS: dict[int, dict] = {
    1: dict(T=353.15, rh_ca=0.50, rh_an=0.50, p_ca=1.5e5, p_an=1.5e5, st_ca=2.0, st_an=1.5),  # var 0
    2: dict(T=323.15, rh_ca=0.50, rh_an=0.50, p_ca=2.5e5, p_an=2.5e5, st_ca=2.0, st_an=1.5),  # var 1
    3: dict(T=353.15, rh_ca=0.30, rh_an=0.30, p_ca=1.5e5, p_an=1.5e5, st_ca=2.5, st_an=1.2),  # var 2
    4: dict(T=353.15, rh_ca=0.30, rh_an=0.30, p_ca=2.5e5, p_an=2.5e5, st_ca=2.5, st_an=2.0),  # var 4
    5: dict(T=353.15, rh_ca=0.30, rh_an=0.50, p_ca=2.3e5, p_an=2.5e5, st_ca=2.0, st_an=1.5),  # var 15
    6: dict(T=363.15, rh_ca=0.50, rh_an=0.50, p_ca=1.5e5, p_an=1.5e5, st_ca=2.5, st_an=1.5),  # var 6
    7: dict(T=323.15, rh_ca=0.80, rh_an=0.80, p_ca=1.5e5, p_an=1.5e5, st_ca=2.0, st_an=1.5),  # var 5
    8: dict(T=353.15, rh_ca=0.80, rh_an=0.80, p_ca=1.5e5, p_an=1.5e5, st_ca=2.5, st_an=2.0),  # var 8
    9: dict(T=353.15, rh_ca=0.80, rh_an=0.80, p_ca=1.5e5, p_an=1.5e5, st_ca=1.5, st_an=1.5),  # var 10
}

# Current density grid [A/cm²] — same for all cases
CURRENT_DENSITIES_A_CM2 = np.array([
    0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
])

CELL_AREA = 25e-4   # m²
FARADAY   = 96485.  # C/mol

BASELINE_DIR = Path(__file__).parent / 'baseline_data'


# ---------------------------------------------------------------------------
# Cell construction
# Adapted from fuel_cell_model_legacy.py → create_fuel_cell, with
# "import marapendi.legacy as mrpd" replaced by "import marapendi as mrpd"
# (the legacy classes are at the top-level at this commit).
# ---------------------------------------------------------------------------

def create_fuel_cell(params: dict) -> mrpd.FuelCell:
    memb = mrpd.PFSA(
        ionomer=mrpd.PFSAIonomer(
            equivalent_weight=params['memb-equiv-weight'],
            dry_density=2000.,
            conductivity_exp=params['memb-cond-exp'],
            conductivity_activation_energy=params['memb-E-act-cond'],
            conductivity_correction=params['memb-cond-correction'],
            reference_water_diffusivity=params['memb-water-diff'],
            reference_water_absorption_coefficient=params['memb-abs-constant'],
            water_diffusivity_activation_energy=params['E-act-memb-diff'],
            water_absorption_activation_energy=params['E-act-memb-abs'],
        ),
        dry_thickness=params['memb-thickness'],
    )

    orr_kinetics = mrpd.ElectrochemicalReaction(
        reference_exchange_current_density=params['i0-c'],
        reaction_order=params['gamma-c'],
        activation_energy=params['E-act-ca'],
        reference_activity=1.01325e5,
        reference_temperature=353.15,
        number_of_electrons=1,
        charge_transfer_coeff=params['alpha-c'],
    )

    liq_model = mrpd.DarcyTransportModel(
        J_function_exponent=params['wet-transition'],
    )

    gdl = {
        side: mrpd.PorousLayer(
            thickness=params['gdl-thickness'],
            contact_angle=params['gdl-theta'],
            effective_gas_diffusion_ratio=params['gdl-eff-diff-ratio'],
            absolute_permeability=params['gdl-abs-perm'],
            porosity=params['gdl-porosity'],
            thermal_conductivity=params['gdl-thermal-cond'],
            two_phase_transport_model=liq_model,
            transport_resistance_model=mrpd.PorousGasDiffusionModel(
                water_saturation_exponent=params['n_s'],
            ),
        ) for side in ['ca', 'an']
    }

    gfc = {
        side: mrpd.FlowChannel(
            height=params['ch-height'],
            width=1e-3,
            n_parallel=1,
            length=21 * 50e-3,
            reactant='o2' if side == 'ca' else 'h2',
            transport_resistance_model=mrpd.ChannelGasResistanceModel(
                sherwood=params['Sh'], B_ch=params['B_ch'],
            ),
        ) for side in ['an', 'ca']
    }

    ionomer = mrpd.PFSAIonomer(
        conductivity_correction=params['ionomer-cond-corr'],
        conductivity_exp=params['ionomer-cond-exp'],
        conductivity_activation_energy=params['ionomer-E-act-cond'],
    )

    return mrpd.FuelCell(
        electrical_resistance=params['elec-resistance'],
        area=CELL_AREA,
        ca=mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(
                ecsa=params['ecsa'],
                platinum_loading=params['pt-loading'],
                ionomer=ionomer,
                catalyst_platinum_weight_percent=params['pt-wt-percent'],
                ionomer_to_carbon_ratio=params['ic-ratio'],
                ionomer_k1=params['ionomer-k1'],
                ionomer_k2=params['ionomer-k2'],
                ionomer_k3=params['ionomer-k3'],
                pore_diameter=params['cl-pore-diameter'],
                omega_PtO=0,
                carbon_agglomerate_radius=params['radius-carbon'],
                thickness=params['pt-loading'] * 2.8e-6 / 0.1e-2,
                absolute_permeability=params['cl-abs-perm'],
                contact_angle=params['cl-theta'],
                thermal_conductivity=params['cl-thermal-cond'],
                reaction=orr_kinetics,
                two_phase_transport_model=liq_model,
                transport_resistance_model=mrpd.PorousGasDiffusionModel(
                    water_saturation_exponent=1.5,
                ),
            ),
            gdl=gdl['ca'],
            has_mpl=False,
            ch=gfc['ca'],
            thermal_contact_resistance=params['tcr'],
        ),
        an=mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(
                ecsa=params['ecsa'],
                platinum_loading=1e-3,
                catalyst_platinum_weight_percent=params['pt-wt-percent'],
                ionomer_to_carbon_ratio=params['ic-ratio'],
                ionomer=ionomer,
                pore_diameter=params['cl-pore-diameter'],
                carbon_agglomerate_radius=params['radius-carbon'],
                thickness=2.8e-6,
                absolute_permeability=params['cl-abs-perm'],
                contact_angle=params['cl-theta'],
                thermal_conductivity=params['cl-thermal-cond'],
                two_phase_transport_model=liq_model,
                transport_resistance_model=mrpd.PorousGasDiffusionModel(
                    water_saturation_exponent=1.5,
                ),
            ),
            has_mpl=False,
            gdl=gdl['an'],
            ch=gfc['an'],
            thermal_contact_resistance=params['tcr'],
        ),
        membrane=memb,
        use_eq_water_content_for_ionomer=True,
    )


# ---------------------------------------------------------------------------
# Operating conditions  (adapted from fuel_cell_model_legacy.py)
# ---------------------------------------------------------------------------

def make_cell_conditions(case_id: int, i_k: float):
    """Return a :class:`CellConditions` for *case_id* at current density *i_k* (A/m²)."""
    c = CASE_CONDITIONS[case_id]
    T = c['T']
    i_max = float(i_k) + 1.

    min_st_ca = 4 / 24.5 / 3600 * 0.21 / (i_max * CELL_AREA / 4 / FARADAY)
    min_st_an = 2 / 24.5 / 3600 * 1.   / (i_max * CELL_AREA / 2 / FARADAY)
    st_ca = max(c['st_ca'], min_st_ca)
    st_an = max(c['st_an'], min_st_an)

    return mrpd.CellConditions(
        current_density=np.array([float(i_k)]),
        cell_temperature=T,
        ca=mrpd.SideConditions(
            inlet_temperature=T,
            inlet_pressure=c['p_ca'],
            outlet_pressure=c['p_ca'],
            dry_o2_mole_fraction=0.21,
            inlet_relative_humidity=c['rh_ca'],
            stoichiometry=st_ca,
        ),
        an=mrpd.SideConditions(
            inlet_temperature=T,
            inlet_pressure=c['p_an'],
            outlet_pressure=c['p_an'],
            dry_h2_mole_fraction=1.0,
            inlet_relative_humidity=c['rh_an'],
            stoichiometry=st_an,
        ),
    )


# ---------------------------------------------------------------------------
# Polarization curve computation
# ---------------------------------------------------------------------------

def compute_polarization_curve(
    params: dict,
    case_id: int,
    current_densities_Am2: np.ndarray,
) -> pd.DataFrame:
    """Build a FuelCell, sweep point-by-point, and return a DataFrame.

    Calling point-by-point gives HFR at every operating point (not just the
    last), which produces a richer non-regression dataset.

    Columns
    -------
    current_density  [A/cm²]
    cell_voltage     [V]
    hfr              [Ω·m²]
    """
    fuel_cell = create_fuel_cell(params)
    model = mrpd.ExplicitSteadyStateModel()

    records = []
    for i_k in current_densities_Am2:
        cond = make_cell_conditions(case_id, float(i_k))
        state = model.set_initial_conditions(fuel_cell, cond)
        state = model.solve(fuel_cell, cond, state)
        fuel_cell.state = state
        hfr = fuel_cell.high_frequency_resistance()
        records.append({
            'current_density': float(i_k) * 1e-4,
            'cell_voltage':    float(np.atleast_1d(state.cell_voltage)[0]),
            'hfr':             float(np.asarray(hfr).flat[0]),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------

def _baseline_path(case_id: int) -> Path:
    return BASELINE_DIR / f'case_{case_id:02d}_baseline.csv'


def save_baseline(df: pd.DataFrame, case_id: int) -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_baseline_path(case_id), index=False)


def load_baseline(case_id: int) -> pd.DataFrame:
    p = _baseline_path(case_id)
    if not p.exists():
        pytest.skip(f'Baseline not found: {p}  — run once to generate it.')
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module', autouse=True)
def generate_baselines_if_missing():
    """Generate baseline CSVs on first run; no-op if all files already exist."""
    missing = [c for c in CASE_CONDITIONS if not _baseline_path(c).exists()]
    if not missing:
        return

    current_Am2 = CURRENT_DENSITIES_A_CM2 * 1e4
    for case_id in missing:
        df = compute_polarization_curve(FULL_PARAMETERS, case_id, current_Am2)
        df.insert(0, 'case_id', case_id)
        save_baseline(df, case_id)


@pytest.mark.parametrize('case_id', sorted(CASE_CONDITIONS))
def test_polarization_curve_matches_baseline(case_id):
    """
    Regenerate the polarization curve for *case_id* and compare against the
    stored baseline row-by-row.

    Tolerances
    ----------
    cell_voltage : ±1 mV  (absolute)
    hfr          : ±0.1 % (relative)
    """
    baseline = load_baseline(case_id)

    current_Am2 = baseline['current_density'].values * 1e4  # A/cm² → A/m²
    result = compute_polarization_curve(FULL_PARAMETERS, case_id, current_Am2)

    assert len(result) == len(baseline), (
        f'Case {case_id}: {len(result)} points computed vs {len(baseline)} in baseline'
    )
    np.testing.assert_allclose(
        result['cell_voltage'].values,
        baseline['cell_voltage'].values,
        atol=1e-3,
        err_msg=f'Case {case_id}: cell voltage deviates by more than 1 mV',
    )
    np.testing.assert_allclose(
        result['hfr'].values,
        baseline['hfr'].values,
        rtol=1e-3,
        err_msg=f'Case {case_id}: HFR deviates by more than 0.1 %',
    )
