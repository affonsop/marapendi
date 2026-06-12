from marapendi.future.state import (
    CatalystLayerState,
    CellSideState,
    CellState,
    FlowChannelState,
    LayerState,
    MembraneState,
)


def test_cell_side_state_default_layers():
    side = CellSideState()
    assert side.porous_layers == [side.gdl, side.cl]
    assert side.layers == [side.ch, side.gdl, side.cl]
    assert isinstance(side.cl, CatalystLayerState)
    assert isinstance(side.gdl, LayerState)
    assert isinstance(side.ch, FlowChannelState)


def test_cell_side_state_without_gdl():
    side = CellSideState(gdl=None)
    assert side.porous_layers == [side.cl]
    assert side.layers == [side.ch, side.cl]


def test_cell_state_layers_and_sides():
    state = CellState()
    assert state.sides == (state.ca, state.an)
    assert state.side_layers == state.ca.layers + state.an.layers
    assert state.layers == state.side_layers + [state.membrane]
    assert isinstance(state.membrane, MembraneState)
