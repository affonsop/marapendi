from marapendi.future.catalyst_layers import PtCCatalystLayer
from marapendi.future.cell import Cell, CellSide
from marapendi.future.flow_channels import FlowChannel
from marapendi.future.membrane import Membrane
from marapendi.future.porous_layers import GasDiffusionLayer, MicroPorousLayer


def test_cell_side_default_layers():
    side = CellSide()
    assert side.porous_layers == [side.cl, side.gdl]
    assert side.layers == [side.cl, side.gdl, side.ch]


def test_cell_side_with_mpl():
    side = CellSide(has_mpl=True)
    assert side.porous_layers == [side.cl, side.mpl, side.gdl]
    assert side.layers == [side.cl, side.mpl, side.gdl, side.ch]


def test_cell_side_without_gdl():
    side = CellSide(has_gdl=False)
    assert side.porous_layers == [side.cl]
    assert side.layers == [side.cl, side.ch]


def test_cell_layers_and_sides():
    cell = Cell()
    assert cell.sides == [cell.ca, cell.an]
    assert cell.layers == cell.an.layers[::-1] + [cell.membrane] + cell.ca.layers
    assert cell.porous_layers == cell.an.porous_layers[::-1] + cell.ca.porous_layers


def test_cell_component_types():
    cell = Cell()
    for side in cell.sides:
        assert isinstance(side.cl, PtCCatalystLayer)
        assert isinstance(side.gdl, GasDiffusionLayer)
        assert isinstance(side.mpl, MicroPorousLayer)
        assert isinstance(side.ch, FlowChannel)
    assert isinstance(cell.membrane, Membrane)
