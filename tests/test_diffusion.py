"""Tests for porous and channel gas-transport resistance models.

Covers:
  - PorousGasResistanceModel (models/porous/diffusion.py)
  - DarcyTransportModel       (models/porous/darcy.py)
  - GasTransportModel         (models/cell/gas_transport.py)
"""
import numpy as np
import pytest
import marapendi as mrpd
from marapendi.cell.state import LayerState, CellSideState, FlowChannelState
from marapendi.cell.gas_transport import GasTransportModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layer_state(temperature=353.15, pressure=1e5, o2=0.21, rh=0., sat=0.):
    state = LayerState(temperature=temperature, pressure=pressure)
    state.non_wetting_saturation = sat
    mrpd.GasModel.set_composition(state, o2, 0., rh, pressure, temperature)
    return state


# ---------------------------------------------------------------------------
# PorousGasResistanceModel
# ---------------------------------------------------------------------------

class TestPorousGasResistanceModel:
    @pytest.fixture
    def model(self):
        return mrpd.PorousGasResistanceModel(water_saturation_exponent=3.0)

    @pytest.fixture
    def gdl(self):
        return mrpd.GasDiffusionLayer(
            thickness=200e-6,
            porosity=0.6,
            effective_gas_diffusion_ratio=0.3,
            pore_diameter=1e6,
        )

    def test_dry_resistance_positive(self, model, gdl):
        state = _make_layer_state(sat=0.)
        R = model.gas_transport_resistance(gdl, state, 'o2')
        assert R > 0

    def test_resistance_increases_with_saturation(self, model, gdl):
        state_dry = _make_layer_state(sat=0.)
        state_wet = _make_layer_state(sat=0.5)
        R_dry = model.gas_transport_resistance(gdl, state_dry, 'o2')
        R_wet = model.gas_transport_resistance(gdl, state_wet, 'o2')
        assert R_wet > R_dry

    def test_saturation_correction_at_zero(self, model):
        correction = model.water_saturation_correction(0.)
        assert np.isclose(correction, 1.0)

    def test_saturation_correction_at_one(self, model):
        correction = model.water_saturation_correction(1.)
        assert correction < 1e-4  # near-zero (clips to 1e-6)

    def test_knudsen_correction_negligible_large_pore(self, model, gdl):
        # pore_diameter=1e6 → Knudsen diffusivity >> molecular, so Knudsen term negligible
        state = _make_layer_state()
        D = mrpd.GasModel.species_diffusion_coefficient(state, 'o2')
        R_mol = model.molecular_diffusion_resistance(gdl, D, water_saturation=0.)
        R_total = model.gas_transport_resistance(gdl, state, 'o2')
        assert np.isclose(R_mol, R_total, rtol=1e-3)

    def test_knudsen_correction_matters_small_pore(self, model):
        # Small pore: Knudsen diffusivity is limiting
        cl = mrpd.PtCCatalystLayer(pore_diameter=40e-9, thickness=10e-6, effective_gas_diffusion_ratio=0.15)
        state = _make_layer_state()
        D = mrpd.GasModel.species_diffusion_coefficient(state, 'o2')
        R_mol = model.molecular_diffusion_resistance(cl, D, water_saturation=0.)
        R_total = model.gas_transport_resistance(cl, state, 'o2')
        assert R_total > R_mol  # Knudsen adds to total resistance


# ---------------------------------------------------------------------------
# DarcyTransportModel
# ---------------------------------------------------------------------------

class TestDarcyTransportModel:
    @pytest.fixture
    def layer(self):
        return mrpd.GasDiffusionLayer(
            thickness=200e-6,
            porosity=0.6,
            contact_angle=120.,
            absolute_permeability=1e-12,
        )

    @pytest.fixture
    def model(self):
        return mrpd.DarcyTransportModel(J_function_exponent=2)

    def test_capillary_pressure_roundtrip(self, model, layer):
        s_in = 0.3
        pc = model.capillary_pressure_from_saturation(layer, s_in)
        s_out = model.saturation_from_capillary_pressure(layer, pc)
        assert np.isclose(s_in, s_out, rtol=1e-6)

    def test_saturation_increases_with_capillary_pressure(self, model, layer):
        pc_low = model.capillary_pressure_from_saturation(layer, 0.1)
        pc_high = model.capillary_pressure_from_saturation(layer, 0.9)
        assert pc_high > pc_low

    def test_saturation_capped_at_one(self, model, layer):
        s = model.saturation_from_capillary_pressure(layer, 1e10)
        assert s <= 1.0

    def test_calculate_equivalent_flow_resistance(self):
        liq = mrpd.DarcyTransportModel()
        gdl = mrpd.GasDiffusionLayer(two_phase_transport_model=liq)
        cl = mrpd.PtCCatalystLayer(two_phase_transport_model=liq)
        side = mrpd.FuelCellSide(gdl=gdl)
        # Just check it runs and returns a number
        result = liq.calculate_equivalent_flow_resistance(side)
        assert result >= 0


# ---------------------------------------------------------------------------
# GasTransportModel integration
# ---------------------------------------------------------------------------

class TestGasTransportModel:
    @pytest.fixture
    def cell_side_and_state(self):
        gdl = mrpd.GasDiffusionLayer(
            thickness=200e-6,
            effective_gas_diffusion_ratio=0.3,
            transport_resistance_model=mrpd.PorousGasResistanceModel(),
        )
        ch = mrpd.FlowChannel(
            width=1e-3, height=1e-3, length=0.1, n_parallel=10, reactant='o2',
        )
        cl = mrpd.PtCCatalystLayer(
            thickness=10e-6,
            effective_gas_diffusion_ratio=0.15,
            pore_diameter=40e-9,
        )
        side = mrpd.FuelCellSide(gdl=gdl, ch=ch, cl=cl, has_mpl=False)

        ch_state = FlowChannelState(temperature=353.15, pressure=1e5, inlet_gas_flow_rate=1e-5)
        mrpd.GasModel.set_composition(ch_state, 0.21, 0., 0.5, 1e5, 353.15)

        gdl_state = _make_layer_state(sat=0.1)
        cl_state = _make_layer_state(sat=0.0)

        from marapendi.cell.state import CatalystLayerState
        cl_state_full = CatalystLayerState(
            temperature=353.15, pressure=1e5,
            non_wetting_saturation=0.0, ionomer_water_content=8.0,
        )
        mrpd.GasModel.set_composition(cl_state_full, 0.21, 0., 0.5, 1e5, 353.15)

        side_state = CellSideState(cl=cl_state_full, gdl=gdl_state, ch=ch_state)
        return side, side_state

    def test_total_resistance_positive(self, cell_side_and_state):
        side, side_state = cell_side_and_state
        model = GasTransportModel()
        R = model.gas_transport_resistance(side, side_state, 'o2')
        assert R > 0

    def test_h2o_resistance_positive(self, cell_side_and_state):
        side, side_state = cell_side_and_state
        model = GasTransportModel()
        R = model.gas_transport_resistance(side, side_state, 'h2o')
        assert R > 0

    def test_resistance_increases_with_saturation(self):
        from marapendi.cell.state import CatalystLayerState
        model = GasTransportModel()
        gdl_dry = _make_layer_state(sat=0.)
        gdl_wet = _make_layer_state(sat=0.5)
        ch_state = FlowChannelState(temperature=353.15, pressure=1e5, inlet_gas_flow_rate=1e-5)
        mrpd.GasModel.set_composition(ch_state, 0.21, 0., 0., 1e5, 353.15)
        cl_state = CatalystLayerState(temperature=353.15, pressure=1e5,
                                       non_wetting_saturation=0., ionomer_water_content=8.)
        mrpd.GasModel.set_composition(cl_state, 0.21, 0., 0., 1e5, 353.15)

        gdl = mrpd.GasDiffusionLayer(thickness=200e-6, effective_gas_diffusion_ratio=0.3)
        ch = mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=10)
        cl = mrpd.PtCCatalystLayer(thickness=10e-6, pore_diameter=40e-9)
        side = mrpd.FuelCellSide(gdl=gdl, ch=ch, cl=cl)

        side_state_dry = CellSideState(cl=cl_state, gdl=gdl_dry, ch=ch_state)
        side_state_wet = CellSideState(cl=cl_state, gdl=gdl_wet, ch=ch_state)

        R_dry = model.gas_transport_resistance(side, side_state_dry, 'o2')
        R_wet = model.gas_transport_resistance(side, side_state_wet, 'o2')
        assert R_wet > R_dry
