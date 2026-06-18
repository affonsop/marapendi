"""Tests for FlowChannel and channel resistance models (components/channel/flow_channels.py)."""
import numpy as np
import pytest
import marapendi as mrpd
from marapendi.cell.state import FlowChannelState
from marapendi.channel.gas_transport_resistance import BakerChannelGasResistanceModel


def _channel_state(temperature=353.15, pressure=1e5, o2=0.21, rh=0., flow_rate=1e-6):
    state = FlowChannelState(temperature=temperature, pressure=pressure,
                              inlet_gas_flow_rate=flow_rate)
    mrpd.GasModel.set_composition(state, o2, 0., rh, pressure, temperature)
    return state


@pytest.fixture
def square_channel():
    return mrpd.FlowChannel(
        width=1e-3,
        height=1e-3,
        length=0.1,
        n_parallel=10,
        reactant='o2',
    )


@pytest.fixture
def baker_channel():
    ch = mrpd.FlowChannel(
        width=0.5e-3,
        height=0.8e-3,
        length=37e-3,
        n_parallel=14,
        reactant='o2',
        transport_resistance_model=BakerChannelGasResistanceModel(A_ch=1.12, B_ch=1.01),
    )
    return ch


class TestFlowChannelGeometry:
    def test_hydraulic_diameter_square(self, square_channel):
        # For a square channel w=h=1mm, D_h = 2wh/(w+h) = 1mm
        assert np.isclose(square_channel.hydraulic_diameter, 1e-3)

    def test_total_flow_section(self, square_channel):
        assert np.isclose(
            square_channel.total_flow_section,
            square_channel.n_parallel * square_channel.width * square_channel.height,
        )

    def test_half_width(self, square_channel):
        assert np.isclose(square_channel.half_width, 0.5e-3)


class TestChannelGasResistanceModel:
    def test_resistance_positive(self, square_channel):
        state = _channel_state()
        R = square_channel.transport_resistance_model.gas_transport_resistance(
            square_channel, state, 'o2', volume_flow_rate=1e-6,
        )
        assert R > 0

    def test_resistance_decreases_with_flow_rate(self, square_channel):
        state = _channel_state()
        R_low = square_channel.transport_resistance_model.gas_transport_resistance(
            square_channel, state, 'o2', volume_flow_rate=1e-7,
        )
        R_high = square_channel.transport_resistance_model.gas_transport_resistance(
            square_channel, state, 'o2', volume_flow_rate=1e-4,
        )
        assert R_low > R_high

    def test_molecular_diffusion_resistance_positive(self, square_channel):
        state = _channel_state()
        D = mrpd.GasModel.species_diffusion_coefficient(state, 'o2')
        R_diff = square_channel.transport_resistance_model.molecular_diffusion_resistance(
            square_channel, D,
        )
        assert R_diff > 0

    def test_convection_resistance_positive(self, square_channel):
        R_conv = square_channel.transport_resistance_model.convection_resistance(
            square_channel, volume_flow_rate=1e-5,
        )
        assert R_conv > 0


class TestBakerChannelGasResistanceModel:
    def test_baker_diffusion_uses_half_width(self, baker_channel):
        state = _channel_state(temperature=353.15, pressure=1e5)
        D = mrpd.GasModel.species_diffusion_coefficient(state, 'o2')
        R_diff = baker_channel.transport_resistance_model.molecular_diffusion_resistance(
            baker_channel, D,
        )
        expected = baker_channel.transport_resistance_model.A_ch * baker_channel.half_width / D
        assert np.isclose(R_diff, expected)

    def test_baker_total_resistance_matches_components(self, baker_channel):
        state = _channel_state(temperature=353.15, pressure=1e5)
        flow_rate = 5e-6
        D = mrpd.GasModel.species_diffusion_coefficient(state, 'o2')
        model = baker_channel.transport_resistance_model
        R_total = model.total_resistance(baker_channel, D, flow_rate)
        R_diff = model.molecular_diffusion_resistance(baker_channel, D)
        R_conv = model.convection_resistance(baker_channel, flow_rate)
        assert np.isclose(R_total, R_diff + R_conv)

    def test_baker_inherits_gas_transport_resistance(self, baker_channel):
        state = _channel_state()
        R = baker_channel.transport_resistance_model.gas_transport_resistance(
            baker_channel, state, 'o2', volume_flow_rate=1e-5,
        )
        assert R > 0
