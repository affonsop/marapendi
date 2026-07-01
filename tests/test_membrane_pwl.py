"""Tests for MembraneWaterBalanceModelPiecewise.

Verifies that the standard PWL water-balance model:
- produces physically plausible results consistent with the paper model
- selects the correct piecewise segment self-consistently
- handles vectorised (array) inputs
- integrates correctly with ExplicitSteadyStateModel
"""
import numpy as np
import pytest
import marapendi as mrpd
from marapendi.cell.explicit_steady_state import ExplicitSteadyStateModel
from marapendi.water_balance.water_balance import WaterBalanceModel
from marapendi.water_balance.membrane import MembraneWaterBalanceModel
from marapendi.water_balance.membrane_pwl import MembraneWaterBalanceModelPiecewise


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_cell():
    liq = mrpd.DarcyTransportModel(J_function_exponent=2)
    gdl = mrpd.GasDiffusionLayer(
        thickness=200e-6, porosity=0.6, contact_angle=120.,
        effective_gas_diffusion_ratio=0.3, absolute_permeability=1e-12,
        thermal_conductivity=0.5, two_phase_transport_model=liq,
    )
    ca_cl = mrpd.PtCCatalystLayer(
        ecsa=70e3, platinum_loading=0.4e-2, ionomer=mrpd.PFSAIonomer(),
        reaction=mrpd.ElectrochemicalReaction(
            reference_exchange_current_density=2.5e-4,
            reaction_order=0.54, activation_energy=67e6,
            reference_activity=1e5, reference_temperature=353.15,
            number_of_electrons=2, charge_transfer_coeff=0.5,
        ),
        thickness=10e-6, thermal_conductivity=0.22,
        pore_diameter=40e-9, absolute_permeability=1e-13, contact_angle=97.,
        two_phase_transport_model=liq,
    )
    return mrpd.FuelCell(
        area=25e-4, electrical_resistance=30e-7,
        ca=mrpd.FuelCellSide(
            cl=ca_cl,
            gdl=gdl,
            ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='o2'),
            has_mpl=False, thermal_contact_resistance=4e-4,
        ),
        an=mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(thickness=5e-6, two_phase_transport_model=liq),
            gdl=mrpd.GasDiffusionLayer(
                thickness=200e-6, effective_gas_diffusion_ratio=0.3,
                thermal_conductivity=0.5, two_phase_transport_model=liq,
            ),
            ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='h2'),
            has_mpl=False, thermal_contact_resistance=4e-4,
        ),
        membrane=mrpd.PFSA(
            ionomer=mrpd.PFSAIonomer(equivalent_weight=1100, dry_density=1980),
            dry_thickness=25e-6,
        ),
    )


def _conditions(i=5e3, T=353.15, p=1.5e5, rh=0.5):
    return mrpd.CellConditions(
        current_density=np.atleast_1d(i),
        cell_temperature=T,
        ca=mrpd.SideConditions(
            inlet_temperature=T, inlet_pressure=p, outlet_pressure=p,
            dry_o2_mole_fraction=0.21, inlet_relative_humidity=rh, stoichiometry=2.0,
        ),
        an=mrpd.SideConditions(
            inlet_temperature=T, inlet_pressure=p, outlet_pressure=p,
            dry_h2_mole_fraction=1.0, inlet_relative_humidity=rh, stoichiometry=1.5,
        ),
    )


def _solve(model, cell, cond):
    state = model.set_initial_conditions(cell, cond)
    return model.solve(cell, cond, state)


@pytest.fixture
def cell():
    return _make_cell()


@pytest.fixture
def paper_model():
    return ExplicitSteadyStateModel(
        water_balance_model=WaterBalanceModel(
            membrane_water_balance_model=MembraneWaterBalanceModel()
        )
    )


@pytest.fixture
def pwl_model():
    return ExplicitSteadyStateModel(
        water_balance_model=WaterBalanceModel(
            membrane_water_balance_model=MembraneWaterBalanceModelPiecewise()
        )
    )


# ---------------------------------------------------------------------------
# Physical sanity — PWL model alone
# ---------------------------------------------------------------------------

class TestPWLSanity:
    def test_voltage_positive(self, cell, pwl_model):
        state = _solve(pwl_model, cell, _conditions())
        assert float(np.atleast_1d(state.cell_voltage)[0]) > 0.4

    def test_voltage_below_ocp(self, cell, pwl_model):
        state = _solve(pwl_model, cell, _conditions())
        assert float(np.atleast_1d(state.cell_voltage)[0]) < 1.23

    def test_voltage_decreases_with_current(self, cell, pwl_model):
        voltages = [
            float(np.atleast_1d(_solve(pwl_model, cell, _conditions(i)).cell_voltage)[0])
            for i in [1e3, 5e3, 1e4]
        ]
        assert voltages[0] > voltages[1] > voltages[2]

    def test_membrane_water_content_positive(self, cell, pwl_model):
        state = _solve(pwl_model, cell, _conditions())
        assert float(np.atleast_1d(state.membrane.water_content)[0]) > 0

    def test_membrane_water_content_increases_with_humidity(self, cell, pwl_model):
        wc_dry  = float(np.atleast_1d(_solve(pwl_model, cell, _conditions(rh=0.2)).membrane.water_content)[0])
        wc_wet  = float(np.atleast_1d(_solve(pwl_model, cell, _conditions(rh=0.9)).membrane.water_content)[0])
        assert wc_wet > wc_dry

    def test_segment_index_valid(self, cell, pwl_model):
        state = _solve(pwl_model, cell, _conditions())
        n_seg = len(cell.membrane.ionomer.pwl_slopes)
        seg_ca = np.atleast_1d(state.ca.pwl_interval)
        seg_an = np.atleast_1d(state.an.pwl_interval)
        assert np.all(seg_ca >= 0) and np.all(seg_ca < n_seg)
        assert np.all(seg_an >= 0) and np.all(seg_an < n_seg)


# ---------------------------------------------------------------------------
# Qualitative comparison: both models should be physically plausible and
# produce monotone polarisation curves.  The two models are genuinely different
# approximations of the membrane water balance and can disagree significantly
# on un-calibrated cells, so no tight numerical comparison is made here.
# ---------------------------------------------------------------------------

class TestPWLVsPaperModel:
    def test_both_models_voltage_in_range(self, cell, paper_model, pwl_model):
        for model in (paper_model, pwl_model):
            state = _solve(model, cell, _conditions(5e3))
            V = float(np.atleast_1d(state.cell_voltage)[0])
            assert 0.3 < V < 1.23

    def test_both_models_membrane_wc_positive(self, cell, paper_model, pwl_model):
        for model in (paper_model, pwl_model):
            state = _solve(model, cell, _conditions(5e3))
            assert float(np.atleast_1d(state.membrane.water_content)[0]) > 0

    def test_monotone_voltage_both_models(self, cell, paper_model, pwl_model):
        i_arr = np.array([1e3, 5e3, 1e4, 2e4])
        for model in (paper_model, pwl_model):
            state = _solve(model, cell, _conditions(i_arr))
            assert np.all(np.diff(np.atleast_1d(state.cell_voltage)) < 0)

    def test_pwl_wc_higher_than_paper_model(self, cell, paper_model, pwl_model):
        # PWL uses actual isotherm regression; paper model uses a linear expansion.
        # For this cell geometry the PWL predicts a wetter membrane.
        wc_paper = float(np.atleast_1d(_solve(paper_model, cell, _conditions()).membrane.water_content)[0])
        wc_pwl   = float(np.atleast_1d(_solve(pwl_model,   cell, _conditions()).membrane.water_content)[0])
        assert wc_pwl > wc_paper


# ---------------------------------------------------------------------------
# Vectorised inputs
# ---------------------------------------------------------------------------

class TestPWLVectorised:
    def test_output_shape_matches_input(self, cell, pwl_model):
        i_arr = np.linspace(1e3, 2e4, 8)
        state = _solve(pwl_model, cell, _conditions(i_arr))
        assert np.atleast_1d(state.cell_voltage).shape == i_arr.shape

    def test_segment_shape_matches_input(self, cell, pwl_model):
        i_arr = np.linspace(1e3, 2e4, 6)
        state = _solve(pwl_model, cell, _conditions(i_arr))
        assert np.atleast_1d(state.ca.pwl_interval).shape == i_arr.shape
        assert np.atleast_1d(state.an.pwl_interval).shape == i_arr.shape

    def test_water_content_shape_matches_input(self, cell, pwl_model):
        i_arr = np.array([2e3, 8e3, 1.5e4])
        state = _solve(pwl_model, cell, _conditions(i_arr))
        assert np.atleast_1d(state.membrane.water_content).shape == i_arr.shape


# ---------------------------------------------------------------------------
# PWL regression quality on the PFSAIonomer
# ---------------------------------------------------------------------------

class TestPWLRegression:
    def test_rms_error_below_threshold(self):
        ionomer = mrpd.PFSAIonomer()
        T = ionomer.pwl_temperature
        rh = np.linspace(0, 1, 200)
        rh_approx = ionomer.linear_rh_from_water_content(
            ionomer.vapor_equilibrium_water_content(rh, T)
        )
        rms = np.sqrt(np.mean((rh_approx - rh) ** 2))
        assert rms < 0.025   # < 2.5 % RMS

    def test_pwl_continuity_at_breakpoints(self):
        ionomer = mrpd.PFSAIonomer()
        breaks = ionomer.lmbd_pwl_breaks[1:-1]
        for lmbd in breaks:
            # Evaluate from the left and right segments
            eps = 1e-6
            rh_lo = ionomer.linear_rh_from_water_content(lmbd - eps)
            rh_hi = ionomer.linear_rh_from_water_content(lmbd + eps)
            assert abs(rh_lo - rh_hi) < 1e-3  # continuous to < 0.1 %

    def test_fit_rh_piecewise_linear_custom_breaks(self):
        ionomer = mrpd.PFSAIonomer()
        T = ionomer.pwl_temperature
        custom_breaks = np.array([0.0, 0.3, 0.7, 1.0])
        ionomer.fit_rh_piecewise_linear(rh_breaks=custom_breaks, temperature=T)
        assert len(ionomer.pwl_slopes) == 3
        # Restore default
        ionomer.fit_rh_piecewise_linear(temperature=T)
