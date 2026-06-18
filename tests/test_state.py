"""Tests for state containers (simulation/state.py)."""
import numpy as np
import pytest
import marapendi as mrpd
from marapendi.cell.state import (
    CellState, CellSideState, LayerState, CatalystLayerState,
    FlowChannelState, MembraneState,
)


class TestLayerState:
    def test_defaults(self):
        state = LayerState()
        assert state.liquid_saturation == 0.
        assert state.non_wetting_saturation == 0.
        assert state.temperature is None

    def test_gas_state_default(self):
        state = LayerState()
        assert state.gas is not None
        assert isinstance(state.gas, mrpd.GasState)


class TestCatalystLayerState:
    def test_inherits_layer_state(self):
        state = CatalystLayerState()
        assert hasattr(state, 'liquid_saturation')
        assert state.ionomer_water_content is None

    def test_set_temperature(self):
        state = CatalystLayerState(temperature=353.15)
        assert state.temperature == 353.15


class TestFlowChannelState:
    def test_defaults(self):
        state = FlowChannelState()
        assert state.inlet_gas_flow_rate == 1e-12

    def test_explicit_fields(self):
        state = FlowChannelState(temperature=353.15, pressure=2e5, inlet_gas_flow_rate=1e-5)
        assert state.temperature == 353.15
        assert state.pressure == 2e5


class TestCellSideState:
    def test_porous_layers_without_mpl(self):
        side_state = CellSideState()
        # Should have cl and gdl, no mpl
        layers = side_state.porous_layers
        assert len(layers) == 2  # gdl + cl

    def test_porous_layers_with_mpl(self):
        side_state = CellSideState(mpl=LayerState())
        layers = side_state.porous_layers
        assert len(layers) == 3  # gdl + mpl + cl

    def test_layers_includes_channel(self):
        side_state = CellSideState()
        all_layers = side_state.layers
        assert side_state.ch in all_layers


class TestCellState:
    def test_defaults(self):
        state = CellState()
        assert state.ca is not None
        assert state.an is not None
        assert state.membrane is not None

    def test_sides_iterator(self):
        state = CellState()
        sides = state.sides
        assert len(sides) == 2
        assert any(s is state.ca for s in sides)
        assert any(s is state.an for s in sides)

    def test_cell_state_fields(self):
        state = CellState(current_density=1e4, temperature=353.15)
        assert state.current_density == 1e4
        assert state.temperature == 353.15

    def test_side_layers(self):
        state = CellState()
        layers = state.side_layers
        assert len(layers) > 0
