import numpy as np

from marapendi.future.conditions import CellOperatingConditions as Conditions
from marapendi.future.model import CellModel, ExplicitSteadyStateModel


def test_cell_model_base_methods_are_no_ops():
    model = CellModel()
    assert model.steady_state_solution(Conditions(current_density=1.)) is None
    assert model.transient_solution(Conditions(current_density=1.)) is None


def test_initial_state_sets_temperature_and_current_density():
    model = ExplicitSteadyStateModel()
    state = model.initial_state(Conditions(current_density=1.))

    assert state.current_density == 1.
    assert state.temperature == 353.15
    assert state.membrane.temperature == 353.15
    for layer in state.side_layers:
        assert layer.temperature == 353.15
        assert layer.liquid_saturation == 0.
        assert layer.non_wetting_saturation == 0.


def test_steady_state_solution_sets_mea_temperature():
    model = ExplicitSteadyStateModel()
    conditions = Conditions(current_density=1., cell_temperature=353.15)
    state = model.steady_state_solution(conditions)

    expected = model.thermal_model.mea_temperature(model.cell, model.initial_state(conditions))
    assert state.membrane.temperature == expected
    assert state.ca.cl.temperature == expected
    assert state.an.cl.temperature == expected
    assert state.membrane.temperature > state.temperature


def test_steady_state_solution_sets_catalyst_layer_gas_concentrations():
    model = ExplicitSteadyStateModel()
    conditions = Conditions(current_density=1., cell_temperature=353.15)
    state = model.steady_state_solution(conditions)

    for side_state in state.sides:
        assert side_state.reactant_transport_resistance > 0
        assert side_state.cl.relative_humidity > 0
        assert np.all(side_state.cl.gas.X >= 0)
        assert np.isclose(np.sum(side_state.cl.gas.X), 1.)

    assert state.ca.cl.gas.X[..., 0] < state.ca.ch.gas.X[..., 0]  # O2 depleted towards the CL
    assert state.an.cl.gas.X[..., 2] < state.an.ch.gas.X[..., 2]  # H2 depleted towards the CL


def test_steady_state_solution_computes_cell_voltage():
    model = ExplicitSteadyStateModel()
    low = model.steady_state_solution(Conditions(current_density=100., cell_temperature=353.15))
    high = model.steady_state_solution(Conditions(current_density=10000., cell_temperature=353.15))

    assert low.E_rev > 0
    assert low.eta_act > 0
    assert low.eta_ohm > 0
    assert 0 < low.cell_voltage < low.E_rev

    # Higher current density -> larger overpotentials and lower cell voltage.
    assert high.eta_act > low.eta_act
    assert high.eta_ohm > low.eta_ohm
    assert high.cell_voltage < low.cell_voltage
