"""Tests for marapendi.components.flow_channels — FlowChannel geometry."""
import numpy as np
import pytest
import marapendi.dynamic as mrpd


@pytest.fixture
def ch():
    return mrpd.FlowChannel(
        width=1e-3,
        height=1e-3,
        length=100e-3,
        n_parallel=14,
        channel_land_ratio=1.,
        bulk_thermal_conductivity=100.,
    )


class TestFlowChannelGeometry:
    def test_hydraulic_diameter_square(self, ch):
        # Square cross-section: D_h = 2*w*h / (w+h) = w = h
        assert ch.hydraulic_diameter == pytest.approx(ch.width, rel=1e-9)

    def test_thickness_equals_height(self, ch):
        assert ch.thickness == pytest.approx(ch.height, rel=1e-9)

    def test_channel_flow_section(self, ch):
        assert ch.channel_flow_section == pytest.approx(ch.width * ch.height, rel=1e-9)

    def test_total_flow_section(self, ch):
        assert ch.total_flow_section == pytest.approx(ch.n_parallel * ch.channel_flow_section, rel=1e-9)

    def test_eps_p_is_one(self, ch):
        assert ch.eps_p == pytest.approx(1.0)


class TestFlowChannelInlets:
    def test_set_fixed_inlet_gas_flow_rate(self, ch):
        ch.set_fixed_inlet_gas_flow_rate(1e-6)
        assert ch.inlet_gas_flow_rate == pytest.approx(1e-6)

    def test_liquid_saturation_zero_without_liquid(self, ch):
        ch.set_fixed_inlet_gas_flow_rate(1e-6)
        assert ch.inlet_liquid_saturation == pytest.approx(0., abs=1e-12)

    def test_liquid_saturation_with_liquid(self, ch):
        ch.set_fixed_inlet_gas_flow_rate(1e-6)
        ch.set_fixed_inlet_liquid_flow_rate(1e-6)
        # Both flows equal → saturation = 0.5
        assert ch.inlet_liquid_saturation == pytest.approx(0.5, rel=1e-6)

    def test_gas_superficial_speed(self, ch):
        Q = 1e-4  # m³/s
        v = ch.gas_superficial_speed(Q)
        assert v == pytest.approx(Q / ch.total_flow_section, rel=1e-9)
