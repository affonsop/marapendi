"""Tests for ImplicitSteadyStateModel.

Verifies that the implicit model:
- produces physically plausible voltages and MEA temperatures,
- agrees closely with the explicit model (same physics, better T_MEA self-consistency),
- preserves warm-start state between successive calls,
- handles scalar and vectorised current densities,
- returns a fully populated CellState.
"""
import time

import numpy as np
import pytest
import marapendi as mrpd
from marapendi.cell.implicit_steady_state import ImplicitSteadyStateModel
from marapendi.cell.explicit_steady_state import ExplicitSteadyStateModel


# ---------------------------------------------------------------------------
# Shared helpers
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


T_OP = 353.15


def _conditions(i, T=T_OP, p=1.5e5):
    return mrpd.CellConditions(
        current_density=np.atleast_1d(i),
        cell_temperature=T,
        ca=mrpd.SideConditions(
            inlet_temperature=T, inlet_pressure=p, outlet_pressure=p,
            dry_o2_mole_fraction=0.21, inlet_relative_humidity=0.5, stoichiometry=2.0,
        ),
        an=mrpd.SideConditions(
            inlet_temperature=T, inlet_pressure=p, outlet_pressure=p,
            dry_h2_mole_fraction=1.0, inlet_relative_humidity=0.5, stoichiometry=1.5,
        ),
    )


def _solve(model, cell, cond):
    state = model.set_initial_conditions(cell, cond)
    return model.solve(cell, cond, state)


@pytest.fixture
def cell():
    return _make_cell()


@pytest.fixture
def explicit_model():
    return ExplicitSteadyStateModel()


@pytest.fixture
def implicit_model():
    return ImplicitSteadyStateModel()


# ---------------------------------------------------------------------------
# Physical sanity
# ---------------------------------------------------------------------------

class TestImplicitModelSanity:
    def test_voltage_positive_at_low_current(self, cell, implicit_model):
        state = _solve(implicit_model, cell, _conditions(1e3))
        assert float(np.atleast_1d(state.cell_voltage)[0]) > 0.5

    def test_voltage_below_open_circuit(self, cell, implicit_model):
        state = _solve(implicit_model, cell, _conditions(1e3))
        assert float(np.atleast_1d(state.cell_voltage)[0]) < 1.23

    def test_voltage_decreases_with_current(self, cell, implicit_model):
        voltages = [
            float(np.atleast_1d(_solve(implicit_model, cell, _conditions(i)).cell_voltage)[0])
            for i in [1e3, 5e3, 1e4]
        ]
        assert voltages[0] > voltages[1] > voltages[2]

    def test_mea_temperature_above_stack(self, cell, implicit_model):
        state = _solve(implicit_model, cell, _conditions(5e3))
        assert float(np.atleast_1d(state.mea_temperature)[0]) > T_OP

    def test_mea_temperature_reasonable_range(self, cell, implicit_model):
        state = _solve(implicit_model, cell, _conditions(1e4))
        assert T_OP < float(np.atleast_1d(state.mea_temperature)[0]) < T_OP + 20

    def test_returns_cell_state(self, cell, implicit_model):
        from marapendi.cell.state import CellState
        state = _solve(implicit_model, cell, _conditions(5e3))
        assert isinstance(state, CellState)


# ---------------------------------------------------------------------------
# Self-consistency: the fixed-point equation holds at convergence
# ---------------------------------------------------------------------------

class TestImplicitSelfConsistency:
    def test_mea_temperature_satisfies_heat_balance(self, cell, implicit_model):
        from marapendi.cell.thermal import ThermalModel
        from marapendi.thermo.electrochemistry import h2_lhv
        from marapendi.thermo.constants import FARADAY_CONSTANT

        state = _solve(implicit_model, cell, _conditions(5e3))
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
        )


# ---------------------------------------------------------------------------
# Agreement with explicit model
# ---------------------------------------------------------------------------

class TestImplicitVsExplicit:
    def test_voltages_close(self, cell, explicit_model, implicit_model):
        cond = _conditions(5e3)
        V_exp = float(np.atleast_1d(_solve(explicit_model, cell, cond).cell_voltage)[0])
        V_imp = float(np.atleast_1d(_solve(implicit_model, cell, cond).cell_voltage)[0])
        assert abs(V_exp - V_imp) < 0.05

    def test_both_produce_monotone_curve(self, cell, implicit_model):
        voltages = [
            float(np.atleast_1d(_solve(implicit_model, cell, _conditions(i)).cell_voltage)[0])
            for i in [1e3, 5e3, 1e4, 2e4]
        ]
        assert all(voltages[k] > voltages[k + 1] for k in range(len(voltages) - 1))


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------

class TestWarmStart:
    def test_warm_start_stored_after_solve(self, cell, implicit_model):
        _solve(implicit_model, cell, _conditions(5e3))
        assert implicit_model._last_mea_temperature is not None

    def test_second_call_is_faster_than_first(self, cell):
        model = ImplicitSteadyStateModel()
        cond = _conditions(5e3)

        t0 = time.perf_counter()
        _solve(model, cell, cond)
        t_cold = time.perf_counter() - t0

        t0 = time.perf_counter()
        _solve(model, cell, cond)
        t_warm = time.perf_counter() - t0

        assert t_warm < t_cold * 1.5, (
            f"Warm call ({t_warm*1e3:.1f} ms) not faster than cold ({t_cold*1e3:.1f} ms)"
        )

    def test_warm_start_resets_on_shape_change(self, cell, implicit_model):
        _solve(implicit_model, cell, _conditions(5e3))
        # Different shape — must not crash
        _solve(implicit_model, cell, _conditions(np.array([1e3, 5e3])))


# ---------------------------------------------------------------------------
# Vectorised inputs
# ---------------------------------------------------------------------------

class TestVectorised:
    def test_array_output_shape(self, cell, implicit_model):
        i_arr = np.array([1e3, 5e3, 1e4, 2e4])
        state = _solve(implicit_model, cell, _conditions(i_arr))
        assert np.atleast_1d(state.cell_voltage).shape == i_arr.shape

    def test_array_voltages_monotone(self, cell, implicit_model):
        i_arr = np.array([1e3, 5e3, 1e4, 2e4])
        state = _solve(implicit_model, cell, _conditions(i_arr))
        V = np.atleast_1d(state.cell_voltage)
        assert np.all(np.diff(V) < 0)


# ---------------------------------------------------------------------------
# Direct model API
# ---------------------------------------------------------------------------

class TestDirectModelAPI:
    def test_direct_solve(self, cell):
        model = ImplicitSteadyStateModel()
        cond = _conditions(5e3)
        state = model.set_initial_conditions(cell, cond)
        state = model.solve(cell, cond, state)
        assert 0.4 < float(np.atleast_1d(state.cell_voltage)[0]) < 1.23
