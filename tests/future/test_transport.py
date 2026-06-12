from marapendi.future.cell import CellSide
from marapendi.future.gas import GasModel
from marapendi.future.state import CellSideState
from marapendi.future.transport import GasTransportModel


def _side_state_at(side: CellSide, temperature: float) -> CellSideState:
    state = CellSideState()
    for layer_state in state.layers:
        layer_state.temperature = temperature
        layer_state.pressure = 1.5e5
        GasModel.set_composition(layer_state, dry_o2_mole_fraction=0.21, dry_h2_mole_fraction=0., relative_humidity=0.5)
    state.cl.ionomer_water_content = 10.
    state.cl.theta_catalyst = 0.1
    return state


def test_gas_transport_resistance_h2o_is_positive():
    side = CellSide()
    state = _side_state_at(side, 353.15)
    model = GasTransportModel()
    resistance = model.gas_transport_resistance(side, state, 'h2o')
    assert resistance > 0


def test_gas_transport_resistance_o2_includes_ionomer_film():
    side = CellSide()
    state = _side_state_at(side, 353.15)
    model = GasTransportModel()

    resistance_o2 = model.gas_transport_resistance(side, state, 'o2')
    resistance_h2o = model.gas_transport_resistance(side, state, 'h2o')
    ionomer_film_resistance = side.cl.o2_ionomer_film_resistance(state.cl)

    assert resistance_o2 > 0
    assert resistance_o2 != resistance_h2o
    assert ionomer_film_resistance > 0
