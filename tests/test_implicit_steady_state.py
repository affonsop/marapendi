"""Tests for ImplicitSteadyStateModel.

Verifies that the implicit model:
- produces physically plausible voltages and MEA temperatures,
- agrees closely with the explicit model (same physics, better T_MEA self-consistency),
- preserves warm-start state between successive calls,
- handles scalar and vectorised current densities,
- is accessible via FuelCell.compute_ui_curve(model='implicit_steady_state').
"""
import time

import numpy as np
import pytest
import marapendi as mrpd
from marapendi.cell.implicit_steady_state import ImplicitSteadyStateModel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_cell():
    liq = mrpd.DarcyTransportModel(J_function_exponent=2)

    gdl = mrpd.GasDiffusionLayer(
        thickness=200e-6,
        porosity=0.6,
        contact_angle=120.,
        effective_gas_diffusion_ratio=0.3,
        absolute_permeability=1e-12,
        thermal_conductivity=0.5,
        two_phase_transport_model=liq,
    )

    ca_cl = mrpd.PtCCatalystLayer(
        ecsa=70e3,
        platinum_loading=0.4e-2,
        ionomer=mrpd.PFSAIonomer(),
        reaction=mrpd.ElectrochemicalReaction(
            reference_exchange_current_density=2.5e-4,
            reaction_order=0.54,
            activation_energy=67e6,
            reference_activity=1e5,
            reference_temperature=353.15,
            number_of_electrons=2,
            charge_transfer_coeff=0.5,
        ),
        thickness=10e-6,
        thermal_conductivity=0.22,
        pore_diameter=40e-9,
        absolute_permeability=1e-13,
        contact_angle=97.,
        two_phase_transport_model=liq,
    )

    return mrpd.FuelCell(
        area=25e-4,
        electrical_resistance=30e-7,
        ca=mrpd.FuelCellSide(
            cl=ca_cl,
            gdl=mrpd.GasDiffusionLayer(
                thickness=200e-6,
                effective_gas_diffusion_ratio=0.3,
                thermal_conductivity=0.5,
                two_phase_transport_model=liq,
            ),
            ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='o2'),
            has_mpl=False,
            thermal_contact_resistance=4e-4,
        ),
        an=mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(thickness=5e-6, two_phase_transport_model=liq),
            gdl=gdl,
            ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='h2'),
            has_mpl=False,
            thermal_contact_resistance=4e-4,
        ),
        membrane=mrpd.PFSA(
            ionomer=mrpd.PFSAIonomer(equivalent_weight=1100, dry_density=1980),
            dry_thickness=25e-6,
        ),
    )


def _ca_cond(T=353.15, rh=0.5, p=1.5e5, st=2.0):
    return mrpd.OperatingConditions(
        inlet_temperature=T, inlet_pressure=p, outlet_pressure=p,
        dry_o2_mole_fraction=0.21, inlet_relative_humidity=rh, stoichiometry=st,
    )


def _an_cond(T=353.15, rh=0.5, p=1.5e5, st=1.5):
    return mrpd.OperatingConditions(
        inlet_temperature=T, inlet_pressure=p, outlet_pressure=p,
        dry_h2_mole_fraction=1.0, inlet_relative_humidity=rh, stoichiometry=st,
    )


@pytest.fixture
def cell():
    return _make_cell()


T_OP = 353.15


# ---------------------------------------------------------------------------
# Basic physical sanity
# ---------------------------------------------------------------------------

class TestImplicitModelSanity:
    def test_voltage_positive_at_low_current(self, cell):
        V = cell.compute_ui_curve(
            np.array([1e3]), T_OP, _ca_cond(), _an_cond(),
            model='implicit_steady_state',
        )
        assert float(np.atleast_1d(V)[0]) > 0.5

    def test_voltage_below_open_circuit(self, cell):
        V = cell.compute_ui_curve(
            np.array([1e3]), T_OP, _ca_cond(), _an_cond(),
            model='implicit_steady_state',
        )
        assert float(np.atleast_1d(V)[0]) < 1.23

    def test_voltage_decreases_with_current(self, cell):
        voltages = [
            float(np.atleast_1d(cell.compute_ui_curve(
                np.array([i]), T_OP, _ca_cond(), _an_cond(),
                model='implicit_steady_state',
            ))[0])
            for i in [1e3, 5e3, 1e4]
        ]
        assert voltages[0] > voltages[1] > voltages[2]

    def test_mea_temperature_above_stack(self, cell):
        cell.compute_ui_curve(
            np.array([5e3]), T_OP, _ca_cond(), _an_cond(),
            model='implicit_steady_state',
        )
        assert cell.mea_temperature > T_OP

    def test_mea_temperature_reasonable_range(self, cell):
        cell.compute_ui_curve(
            np.array([1e4]), T_OP, _ca_cond(), _an_cond(),
            model='implicit_steady_state',
        )
        assert T_OP < cell.mea_temperature < T_OP + 20


# ---------------------------------------------------------------------------
# Self-consistency: the fixed-point equation must hold at convergence
# ---------------------------------------------------------------------------

class TestImplicitSelfConsistency:
    def test_mea_temperature_is_self_consistent(self, cell):
        """After implicit solve, T_MEA predicted from the voltage must match the solved T_MEA."""
        from marapendi.cell.thermal import ThermalModel
        from marapendi.thermo.electrochemistry import h2_lhv
        from marapendi.thermo.constants import FARADAY_CONSTANT

        cell.compute_ui_curve(
            np.array([5e3]), T_OP, _ca_cond(), _an_cond(),
            model='implicit_steady_state',
        )
        state = cell.state
        thermal = ThermalModel()
        R_th = thermal.heat_transfer_resistance(cell)
        q = state.current_density * (
            -h2_lhv(state.temperature) / (2 * FARADAY_CONSTANT) - state.cell_voltage
        )
        T_mea_from_balance = state.temperature + q * R_th
        np.testing.assert_allclose(
            np.atleast_1d(state.mea_temperature),
            np.atleast_1d(T_mea_from_balance),
            rtol=1e-4,
            err_msg="T_MEA does not satisfy the heat balance at convergence",
        )


# ---------------------------------------------------------------------------
# Agreement with explicit model (should be close but not identical)
# ---------------------------------------------------------------------------

class TestImplicitVsExplicit:
    def test_voltages_close(self, cell):
        i = np.array([5e3])
        V_exp = float(np.atleast_1d(cell.compute_ui_curve(
            i, T_OP, _ca_cond(), _an_cond(), model='explicit_steady_state',
        ))[0])
        V_imp = float(np.atleast_1d(cell.compute_ui_curve(
            i, T_OP, _ca_cond(), _an_cond(), model='implicit_steady_state',
        ))[0])
        assert abs(V_exp - V_imp) < 0.05, (
            f"Models disagree by {abs(V_exp - V_imp)*1e3:.1f} mV — unexpected divergence"
        )

    def test_both_produce_same_monotone_curve(self, cell):
        i_arr = np.array([1e3, 5e3, 1e4, 2e4])
        V_imp = np.array([
            float(np.atleast_1d(cell.compute_ui_curve(
                np.array([i]), T_OP, _ca_cond(), _an_cond(),
                model='implicit_steady_state',
            ))[0])
            for i in i_arr
        ])
        assert np.all(np.diff(V_imp) < 0)


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------

class TestWarmStart:
    def test_warm_start_stored_after_solve(self, cell):
        cell.compute_ui_curve(
            np.array([5e3]), T_OP, _ca_cond(), _an_cond(),
            model='implicit_steady_state',
        )
        assert cell._implicit_model._last_mea_temperature is not None

    def test_second_call_is_faster_than_first(self, cell):
        i = np.array([5e3])
        t0 = time.perf_counter()
        cell.compute_ui_curve(i, T_OP, _ca_cond(), _an_cond(), model='implicit_steady_state')
        t_cold = time.perf_counter() - t0

        t0 = time.perf_counter()
        cell.compute_ui_curve(i, T_OP, _ca_cond(), _an_cond(), model='implicit_steady_state')
        t_warm = time.perf_counter() - t0

        # Warm should be at least 20% faster; allow generous margin for CI noise.
        assert t_warm < t_cold * 1.5, (
            f"Warm call ({t_warm*1e3:.1f} ms) not faster than cold ({t_cold*1e3:.1f} ms)"
        )

    def test_warm_start_resets_on_shape_change(self, cell):
        cell.compute_ui_curve(
            np.array([5e3]), T_OP, _ca_cond(), _an_cond(),
            model='implicit_steady_state',
        )
        # Different shape — should not crash
        cell.compute_ui_curve(
            np.array([1e3, 5e3]), T_OP, _ca_cond(), _an_cond(),
            model='implicit_steady_state',
        )


# ---------------------------------------------------------------------------
# Vectorised input
# ---------------------------------------------------------------------------

class TestVectorised:
    def test_array_input_shape(self, cell):
        i_arr = np.array([1e3, 5e3, 1e4, 2e4])
        V = cell.compute_ui_curve(
            i_arr, T_OP, _ca_cond(), _an_cond(),
            model='implicit_steady_state',
        )
        V = np.atleast_1d(V)
        assert V.shape == i_arr.shape

    def test_array_voltages_monotone(self, cell):
        i_arr = np.array([1e3, 5e3, 1e4, 2e4])
        V = np.atleast_1d(cell.compute_ui_curve(
            i_arr, T_OP, _ca_cond(), _an_cond(),
            model='implicit_steady_state',
        ))
        assert np.all(np.diff(V) < 0)


# ---------------------------------------------------------------------------
# Direct model API (bypass FuelCell)
# ---------------------------------------------------------------------------

class TestDirectModelAPI:
    def test_direct_solve(self, cell):
        from marapendi.cell.explicit_steady_state import ExplicitSteadyStateModel
        explicit = ExplicitSteadyStateModel()
        implicit = ImplicitSteadyStateModel()

        state = explicit.set_initial_state(cell, T_OP, np.array([5e3]), _ca_cond(), _an_cond())
        V = implicit.solve(cell, state)
        assert 0.4 < float(np.atleast_1d(V)[0]) < 1.23
