from marapendi.future.conditions import CellOperatingConditions as Conditions
from marapendi.future.model import ExplicitSteadyStateModel


def test_solve_water_balance_sets_membrane_water_content():
    model = ExplicitSteadyStateModel()
    state = model.steady_state_solution(Conditions(current_density=1.))

    assert state.membrane.water_content_profile is not None
    assert state.membrane.water_content > 0
    assert state.membrane.peclet_number is not None


def test_solve_water_balance_sets_ionomer_water_content():
    model = ExplicitSteadyStateModel()
    state = model.steady_state_solution(Conditions(current_density=1.))

    for side_state in state.sides:
        assert side_state.cl.eq_water_content is not None
        assert side_state.cl.ionomer_water_content == side_state.cl.eq_water_content


def test_membrane_proton_resistance_from_water_content_profile():
    model = ExplicitSteadyStateModel()
    state = model.steady_state_solution(Conditions(current_density=1.))

    resistance = model.cell.membrane.proton_resistance(
        state.membrane.water_content_profile, state.membrane.temperature,
    )
    assert resistance > 0
