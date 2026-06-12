"""Tests for marapendi.models.flowfields — GasFlowFieldModel."""
import numpy as np
import pytest
from types import SimpleNamespace

import marapendi.dynamic as mrpd
from marapendi.dynamic.models.flowfields import GasFlowFieldModel


T = 353.15
P = 1.5e5


@pytest.fixture
def ff():
    return GasFlowFieldModel()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _channel(p_g_val=1.6e5, backpressure=P, length=0.1, d_h=1e-3,
             fRe=56.0, mu_g=2e-5, mu_l=3.5e-4, rho_l=971.0,
             s=0.05, cg_k=None, section=1e-6):
    """Return (state, gfc, conditions) mocks for a single-layer channel."""
    ix = 0  # single-layer index

    state = SimpleNamespace(
        p_g=np.array([p_g_val]),
        mu_g=np.array([mu_g]),
        mu_l=np.array([mu_l]),
        rho_l=np.array([rho_l]),
        s=np.array([s]),
        cg_k=np.array([[1., 4., 0., 0.1]]) if cg_k is None else np.array([cg_k]),
    )
    gfc = SimpleNamespace(
        ix=ix,
        length=length,
        hydraulic_diameter=d_h,
        fRe=fRe,
        total_flow_section=section,
        total_volume=section * length,
        superficial_velocity=lambda vol_flow, _s=section: vol_flow / _s,
    )
    conditions = SimpleNamespace(
        backpressure=backpressure,
        temperature=T,
        inlet_gas_molar_flow_rate=5e-5,
        inlet_gas_molar_flow_rates=np.array([1e-5, 3e-5, 0., 1e-6]),
        inlet_liquid_mass_flow_rate=0.0,
    )
    return state, gfc, conditions


# ─── gas_to_liquid_slip_ratio ─────────────────────────────────────────────────

class TestGasToLiquidSlipRatio:
    def test_equal_viscosities_half_saturation(self, ff):
        # μ_g = μ_l, s_l = 0.5 → (0.5/0.5)³ * 1 = 1
        ratio = ff.gas_to_liquid_slip_ratio(mu_g=1e-3, mu_l=1e-3, s_l=0.5)
        assert ratio == pytest.approx(1.0)

    def test_positive(self, ff):
        ratio = ff.gas_to_liquid_slip_ratio(mu_g=2e-5, mu_l=3.5e-4, s_l=0.1)
        assert ratio > 0

    def test_increases_with_liquid_saturation(self, ff):
        r_low  = ff.gas_to_liquid_slip_ratio(2e-5, 3.5e-4, 0.1)
        r_high = ff.gas_to_liquid_slip_ratio(2e-5, 3.5e-4, 0.5)
        assert r_high > r_low

    def test_increases_with_gas_viscosity(self, ff):
        r_low  = ff.gas_to_liquid_slip_ratio(mu_g=1e-5, mu_l=3.5e-4, s_l=0.3)
        r_high = ff.gas_to_liquid_slip_ratio(mu_g=4e-5, mu_l=3.5e-4, s_l=0.3)
        assert r_high > r_low

    def test_decreases_with_liquid_viscosity(self, ff):
        r_low  = ff.gas_to_liquid_slip_ratio(mu_g=2e-5, mu_l=1e-3, s_l=0.3)
        r_high = ff.gas_to_liquid_slip_ratio(mu_g=2e-5, mu_l=1e-4, s_l=0.3)
        assert r_high > r_low  # lower mu_l → higher ratio


# ─── calculate_outlet_flows ───────────────────────────────────────────────────

class TestCalculateOutletFlows:
    def test_positive_flows_when_channel_above_backpressure(self, ff):
        state, gfc, cond = _channel(p_g_val=1.6e5, backpressure=P)
        n_dot, m_dot = ff.calculate_outlet_flows(state, gfc, cond)
        assert np.all(n_dot >= 0)
        assert float(m_dot) >= 0

    def test_zero_flow_at_equal_pressures(self, ff):
        state, gfc, cond = _channel(p_g_val=P, backpressure=P)
        n_dot, m_dot = ff.calculate_outlet_flows(state, gfc, cond)
        assert np.all(n_dot == pytest.approx(0.0, abs=1e-20))
        assert float(m_dot) == pytest.approx(0.0, abs=1e-20)

    def test_outlet_increases_with_pressure_difference(self, ff):
        _, _, cond = _channel()
        state1, gfc1, _ = _channel(p_g_val=1.55e5, backpressure=P)
        state2, gfc2, _ = _channel(p_g_val=1.70e5, backpressure=P)
        n1, _ = ff.calculate_outlet_flows(state1, gfc1, cond)
        n2, _ = ff.calculate_outlet_flows(state2, gfc2, cond)
        # Only check species with non-zero concentration; H2 (index 2) is absent
        nonzero = state1.cg_k[gfc1.ix] > 0
        assert np.all(n2[nonzero] > n1[nonzero])

    def test_shape(self, ff):
        state, gfc, cond = _channel()
        n_dot, m_dot = ff.calculate_outlet_flows(state, gfc, cond)
        assert n_dot.shape == (4,)   # one value per species
        assert np.ndim(m_dot) == 0 or m_dot.shape == ()

    def test_liquid_outlet_zero_at_zero_saturation(self, ff):
        state, gfc, cond = _channel(s=0.0)
        _, m_dot = ff.calculate_outlet_flows(state, gfc, cond)
        assert float(m_dot) == pytest.approx(0.0, abs=1e-20)


# ─── calculate_inlet_gas_pressure ────────────────────────────────────────────

class TestCalculateInletGasPressure:
    def test_inlet_pressure_exceeds_channel_pressure(self, ff):
        state, gfc, cond = _channel(p_g_val=P)
        p_in = ff.calculate_inlet_gas_pressure(state, gfc, cond)
        assert p_in >= state.p_g[gfc.ix]

    def test_inlet_pressure_stored_on_conditions(self, ff):
        state, gfc, cond = _channel(p_g_val=P)
        p_in = ff.calculate_inlet_gas_pressure(state, gfc, cond)
        assert cond.inlet_pressure == pytest.approx(p_in)

    def test_higher_flow_gives_higher_inlet_pressure(self, ff):
        state, gfc, cond_low = _channel(p_g_val=P)
        _, _, cond_high = _channel(p_g_val=P)
        cond_low.inlet_gas_molar_flow_rate = 1e-5
        cond_high.inlet_gas_molar_flow_rate = 1e-4
        p_low  = ff.calculate_inlet_gas_pressure(state, gfc, cond_low)
        p_high = ff.calculate_inlet_gas_pressure(state, gfc, cond_high)
        assert p_high > p_low
