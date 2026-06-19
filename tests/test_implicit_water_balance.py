"""Tests for ImplicitWaterBalanceModel.

Verifies that the implicit membrane water balance:
- produces physically plausible voltages and MEA temperatures,
- satisfies the self-consistency condition J_mb = F(λ_eq(rh_cl(J_mb))),
- agrees closely with the explicit model at low current (negligible crossover),
- diverges from the explicit model at high current (significant crossover flux),
- handles vectorised current densities,
- works with ImplicitSteadyStateModel.
"""
import numpy as np
import pytest
import marapendi as mrpd
from marapendi.cell.water_balance import ImplicitWaterBalanceModel, MembraneWaterBalanceModel
from marapendi.cell.explicit_steady_state import ExplicitSteadyStateModel
from marapendi.cell.implicit_steady_state import ImplicitSteadyStateModel


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
            gdl=mrpd.GasDiffusionLayer(
                thickness=200e-6, effective_gas_diffusion_ratio=0.3,
                thermal_conductivity=0.5, two_phase_transport_model=liq,
            ),
            ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='o2'),
            has_mpl=False, thermal_contact_resistance=4e-4,
        ),
        an=mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(thickness=5e-6, two_phase_transport_model=liq),
            gdl=gdl,
            ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='h2'),
            has_mpl=False, thermal_contact_resistance=4e-4,
        ),
        membrane=mrpd.PFSA(
            ionomer=mrpd.PFSAIonomer(equivalent_weight=1100, dry_density=1980),
            dry_thickness=25e-6,
        ),
    )


T_OP = 353.15


def _conditions(i, T=T_OP, p=1.5e5, rh=0.5):
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


def _solve(cell, cond, steady_model):
    state = steady_model.set_initial_conditions(cell, cond)
    return steady_model.solve(cell, cond, state)


@pytest.fixture
def cell():
    return _make_cell()


@pytest.fixture
def exp_model():
    return ExplicitSteadyStateModel(
        water_balance_model=MembraneWaterBalanceModel()
    )


@pytest.fixture
def imp_wb_exp_ss(cell):
    return ExplicitSteadyStateModel(
        water_balance_model=ImplicitWaterBalanceModel()
    )


@pytest.fixture
def imp_wb_imp_ss(cell):
    return ImplicitSteadyStateModel(
        water_balance_model=ImplicitWaterBalanceModel()
    )


# ---------------------------------------------------------------------------
# Physical sanity
# ---------------------------------------------------------------------------

class TestImplicitWaterBalanceSanity:
    def test_voltage_positive(self, cell, imp_wb_exp_ss):
        state = _solve(cell, _conditions(5e3), imp_wb_exp_ss)
        assert float(np.atleast_1d(state.cell_voltage)[0]) > 0.4

    def test_voltage_below_ocp(self, cell, imp_wb_exp_ss):
        state = _solve(cell, _conditions(5e3), imp_wb_exp_ss)
        assert float(np.atleast_1d(state.cell_voltage)[0]) < 1.23

    def test_voltage_decreases_with_current(self, cell, imp_wb_exp_ss):
        voltages = [
            float(np.atleast_1d(_solve(cell, _conditions(i), imp_wb_exp_ss).cell_voltage)[0])
            for i in [1e3, 5e3, 1e4]
        ]
        assert voltages[0] > voltages[1] > voltages[2]

    def test_mea_temperature_above_stack(self, cell, imp_wb_exp_ss):
        state = _solve(cell, _conditions(5e3), imp_wb_exp_ss)
        assert float(np.atleast_1d(state.mea_temperature)[0]) > T_OP

    def test_liquid_flux_non_negative(self, cell, imp_wb_exp_ss):
        state = _solve(cell, _conditions(1e4), imp_wb_exp_ss)
        assert np.all(np.atleast_1d(state.ca.liquid_flux) >= -1e-12)

    def test_membrane_water_flux_finite(self, cell, imp_wb_exp_ss):
        state = _solve(cell, _conditions(5e3), imp_wb_exp_ss)
        assert np.all(np.isfinite(np.atleast_1d(state.ca.membrane_water_flux)))


# ---------------------------------------------------------------------------
# Self-consistency: solved J_mb satisfies the fixed-point
# ---------------------------------------------------------------------------

class TestSelfConsistency:
    def test_membrane_flux_consistent_with_water_content(self, cell, imp_wb_exp_ss):
        """After convergence, the cathode membrane flux must equal the flux
        computed from the converged water content profile."""
        from marapendi.cell.water_balance import ImplicitWaterBalanceModel
        state = _solve(cell, _conditions(5e3), imp_wb_exp_ss)
        wb = imp_wb_exp_ss.water_balance_model
        J_mb_recomputed = wb.calculate_cathode_membrane_flux(state)
        np.testing.assert_allclose(
            np.atleast_1d(state.ca.membrane_water_flux),
            np.atleast_1d(J_mb_recomputed),
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# Agreement / divergence with explicit water balance
# ---------------------------------------------------------------------------

class TestVsExplicitWaterBalance:
    def test_voltages_close_at_low_current(self, cell, exp_model, imp_wb_exp_ss):
        V_exp = float(np.atleast_1d(_solve(cell, _conditions(1e3), exp_model).cell_voltage)[0])
        V_imp = float(np.atleast_1d(_solve(cell, _conditions(1e3), imp_wb_exp_ss).cell_voltage)[0])
        assert abs(V_exp - V_imp) < 0.05

    def test_membrane_flux_differs_at_high_current(self, cell, exp_model, imp_wb_exp_ss):
        """The two models should give different membrane fluxes at high current
        where the crossover feedback is significant."""
        J_exp = float(np.atleast_1d(
            _solve(cell, _conditions(2e4), exp_model).ca.membrane_water_flux)[0])
        J_imp = float(np.atleast_1d(
            _solve(cell, _conditions(2e4), imp_wb_exp_ss).ca.membrane_water_flux)[0])
        # They should disagree by more than numerical noise
        assert abs(J_exp - J_imp) > 1e-8


# ---------------------------------------------------------------------------
# Vectorised inputs
# ---------------------------------------------------------------------------

class TestVectorised:
    def test_array_output_shape(self, cell, imp_wb_exp_ss):
        i_arr = np.array([1e3, 5e3, 1e4, 2e4])
        state = _solve(cell, _conditions(i_arr), imp_wb_exp_ss)
        assert np.atleast_1d(state.cell_voltage).shape == i_arr.shape

    def test_array_voltages_monotone(self, cell, imp_wb_exp_ss):
        i_arr = np.array([1e3, 5e3, 1e4, 2e4])
        state = _solve(cell, _conditions(i_arr), imp_wb_exp_ss)
        V = np.atleast_1d(state.cell_voltage)
        assert np.all(np.diff(V) < 0)


# ---------------------------------------------------------------------------
# Combined with ImplicitSteadyStateModel
# ---------------------------------------------------------------------------

class TestWithImplicitSteadyState:
    def test_voltage_positive(self, cell, imp_wb_imp_ss):
        state = _solve(cell, _conditions(5e3), imp_wb_imp_ss)
        assert float(np.atleast_1d(state.cell_voltage)[0]) > 0.4

    def test_mea_temperature_self_consistent(self, cell, imp_wb_imp_ss):
        """T_MEA from the heat balance must match the stored value."""
        from marapendi.thermo.electrochemistry import h2_lhv
        from marapendi.thermo.constants import FARADAY_CONSTANT
        from marapendi.cell.thermal import ThermalModel

        state = _solve(cell, _conditions(5e3), imp_wb_imp_ss)
        R_th = ThermalModel().heat_transfer_resistance(cell)
        q = state.current_density * (
            -h2_lhv(state.temperature) / (2 * FARADAY_CONSTANT) - state.cell_voltage
        )
        T_mea_balance = state.temperature + q * R_th
        np.testing.assert_allclose(
            np.atleast_1d(state.mea_temperature),
            np.atleast_1d(T_mea_balance),
            rtol=1e-3,
        )

    def test_array_shape(self, cell, imp_wb_imp_ss):
        i_arr = np.linspace(1e3, 2e4, 10)
        state = _solve(cell, _conditions(i_arr), imp_wb_imp_ss)
        assert np.atleast_1d(state.cell_voltage).shape == i_arr.shape
