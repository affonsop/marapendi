"""Tests for PorousLayer and subclasses (components/porous/porous_layers.py)."""
import numpy as np
import pytest
import marapendi as mrpd
from marapendi.simulation.state import LayerState


@pytest.fixture
def gdl():
    return mrpd.GasDiffusionLayer(
        thickness=200e-6,
        porosity=0.6,
        contact_angle=120.,
        absolute_permeability=1e-12,
        effective_gas_diffusion_ratio=0.3,
    )


@pytest.fixture
def mpl():
    return mrpd.MicroPorousLayer(
        thickness=30e-6,
        porosity=0.4,
        contact_angle=130.,
        absolute_permeability=1e-13,
    )


@pytest.fixture
def layer_state():
    state = LayerState(temperature=353.15, pressure=1e5)
    mrpd.GasModel.set_composition(state, 0.21, 0., 0.5, 1e5, 353.15)
    state.non_wetting_saturation = 0.1
    return state


class TestPorousLayerGeometry:
    def test_gdl_defaults(self, gdl):
        assert gdl.thickness == 200e-6
        assert gdl.porosity == 0.6
        assert gdl.contact_angle == 120.

    def test_mpl_defaults(self, mpl):
        assert mpl.thickness == 30e-6
        assert mpl.porosity == 0.4

    def test_thermal_resistance(self, gdl):
        expected = gdl.thickness / gdl.thermal_conductivity
        assert np.isclose(gdl.thermal_resistance, expected)

    def test_breakthrough_pressure_positive(self, gdl):
        assert gdl.breakthrough_pressure > 0

    def test_saturation_flow_resistance_positive(self, gdl):
        assert gdl.saturation_flow_resistance > 0


class TestPorousLayerCapillarity:
    def test_capillary_roundtrip(self, gdl):
        """capillary_pressure → saturation → capillary_pressure should be close to identity."""
        s_in = 0.3
        pc = gdl.two_phase_transport_model.capillary_pressure_from_saturation(gdl, s_in)
        s_out = gdl.two_phase_transport_model.saturation_from_capillary_pressure(gdl, pc)
        assert np.isclose(s_in, s_out, rtol=1e-6)

    def test_saturation_increases_with_capillary_pressure(self, gdl):
        pc_low = gdl.two_phase_transport_model.capillary_pressure_from_saturation(gdl, 0.1)
        pc_high = gdl.two_phase_transport_model.capillary_pressure_from_saturation(gdl, 0.8)
        assert pc_high > pc_low


class TestPorousGasDiffusionModel:
    def test_gas_resistance_positive(self, gdl, layer_state):
        R = gdl.transport_resistance_model.gas_transport_resistance(gdl, layer_state, 'o2')
        assert R > 0

    def test_resistance_increases_with_saturation(self, gdl):
        s_low = LayerState(temperature=353.15, pressure=1e5)
        s_low.non_wetting_saturation = 0.0
        s_high = LayerState(temperature=353.15, pressure=1e5)
        s_high.non_wetting_saturation = 0.5
        for s in (s_low, s_high):
            mrpd.GasModel.set_composition(s, 0.21, 0., 0., 1e5, 353.15)

        R_low = gdl.transport_resistance_model.gas_transport_resistance(gdl, s_low, 'o2')
        R_high = gdl.transport_resistance_model.gas_transport_resistance(gdl, s_high, 'o2')
        assert R_high > R_low

    def test_resistance_thicker_layer_larger(self):
        thin = mrpd.GasDiffusionLayer(thickness=100e-6, effective_gas_diffusion_ratio=0.3)
        thick = mrpd.GasDiffusionLayer(thickness=300e-6, effective_gas_diffusion_ratio=0.3)
        state = LayerState(temperature=353.15, pressure=1e5)
        state.non_wetting_saturation = 0.
        mrpd.GasModel.set_composition(state, 0.21, 0., 0., 1e5, 353.15)
        R_thin = thin.transport_resistance_model.gas_transport_resistance(thin, state, 'o2')
        R_thick = thick.transport_resistance_model.gas_transport_resistance(thick, state, 'o2')
        assert R_thick > R_thin
