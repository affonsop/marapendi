from marapendi.cell import Cell, CellSide
from marapendi.state import CellState
from marapendi.thermal import ThermalModel


def test_side_heat_transfer_resistance_excludes_catalyst_layer():
    side = CellSide()
    model = ThermalModel()
    expected = sum(
        layer.thermal_resistance() for layer in side.porous_layers if layer is not side.cl
    ) + side.thermal_contact_resistance
    assert model.side_heat_transfer_resistance(side) == expected


def test_heat_transfer_resistance_combines_sides_in_parallel():
    cell = Cell()
    model = ThermalModel()
    expected = 1. / sum(1. / model.side_heat_transfer_resistance(side) for side in cell.sides)
    assert model.heat_transfer_resistance(cell) == expected


def test_mea_temperature_above_bulk_temperature():
    cell = Cell()
    model = ThermalModel()
    state = CellState(current_density=1., temperature=353.15)
    mea_temperature = model.mea_temperature(cell, state)
    assert mea_temperature > state.temperature


def test_set_mea_temperature():
    state = CellState()
    ThermalModel().set_mea_temperature(360., state)
    assert state.membrane.temperature == 360.
    assert state.ca.cl.temperature == 360.
    assert state.an.cl.temperature == 360.
