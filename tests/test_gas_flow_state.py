"""Tests for GasFlowState (simulation/state.py).

Checks:
- Importing marapendi does not corrupt numpy.ndarray (regression guard for a
  previous broken dataclass field annotation on this class).
- GasFlowState.to_side_conditions() reproduces the same steady-state
  solution as specifying an equivalent stoichiometry directly. The
  GasFlowState is built from the channel flow ExplicitSteadyStateModel
  itself already computed from the stoichiometry, so both paths are
  checked against the same ground truth rather than two independent
  hand-derived numbers.
"""
import numpy as np
import pytest
import marapendi as mrpd
from marapendi.simulation.state import GasFlowState
from marapendi.models.thermo.gas import GasModel, index_o2, index_n2, index_h2, index_h2ov
from marapendi.models.thermo.water import water_molar_volume
from marapendi.models.thermo.constants import FARADAY_CONSTANT
from marapendi.models.base.explicit_steady_state import ExplicitSteadyStateModel
from marapendi.models.base.transient import TransientModel
from marapendi.interop.simulink_bridge import build_default_cell


def test_import_does_not_corrupt_numpy_ndarray():
    """Regression guard: GasFlowState previously used `x = np.ndarray = field(...)`
    instead of a type annotation, which silently reassigned numpy.ndarray
    itself (and broke the whole package import) as a side effect of
    defining the dataclass."""
    assert np.ndarray.__module__ == 'numpy'
    assert isinstance(np.array([1.]), np.ndarray)


T_OP = 344.15
I_OP = 1e4  # A/m2
P_OP = 1.5e5  # Pa, same on inlet and outlet -> average_pressure == P_OP exactly


def _conditions(i=I_OP, T=T_OP, p=P_OP, rh=0.5):
    return mrpd.CellConditions(
        current_density=np.atleast_1d(i),
        cell_temperature=T,
        ca=mrpd.SideConditions(
            inlet_temperature=T, inlet_pressure=p, outlet_pressure=p,
            dry_o2_mole_fraction=0.21, inlet_relative_humidity=rh, stoichiometry=2.0,
        ),
        an=mrpd.SideConditions(
            inlet_temperature=T, inlet_pressure=p, outlet_pressure=p,
            dry_o2_mole_fraction=0., dry_h2_mole_fraction=1.0,
            inlet_relative_humidity=rh, stoichiometry=1.5,
        ),
    )


def _gas_flow_state_from_channel(ch_state, temperature) -> GasFlowState:
    """Build a GasFlowState carrying the same total inlet flow as an
    already-solved FlowChannelState (computed internally from a
    stoichiometry spec by ExplicitSteadyStateModel._set_flow_rates)."""
    concentration = GasModel.concentration(ch_state)
    total_gas_molar_flow_rate = ch_state.inlet_gas_flow_rate * concentration
    return GasFlowState(
        temperature=temperature,
        pressure=float(np.asarray(ch_state.pressure).reshape(-1)[0]),
        gas_species_molar_flow_rates=np.asarray(ch_state.gas.X).reshape(-1) * total_gas_molar_flow_rate,
        liquid_molar_flow_rate=float(np.asarray(ch_state.inlet_liquid_flow_rate).reshape(-1)[0])
        / water_molar_volume(temperature),
    )


class TestGasFlowStateRoundTrip:
    """GasFlowState.to_side_conditions() should reproduce a stoichiometry-based solve."""

    def test_matches_stoichiometry_specification(self):
        cell = build_default_cell()
        conditions = _conditions()
        model = ExplicitSteadyStateModel()

        state_stoich = model.set_initial_conditions(cell, conditions)
        state_stoich = model.solve(cell, conditions, state_stoich)

        ca_flow = _gas_flow_state_from_channel(state_stoich.ca.ch, T_OP)
        an_flow = _gas_flow_state_from_channel(state_stoich.an.ch, T_OP)

        conditions_from_flow = mrpd.CellConditions(
            current_density=conditions.current_density,
            cell_temperature=conditions.cell_temperature,
            ca=ca_flow.to_side_conditions(),
            an=an_flow.to_side_conditions(),
        )

        state_flow = model.set_initial_conditions(cell, conditions_from_flow)
        state_flow = model.solve(cell, conditions_from_flow, state_flow)

        assert state_flow.cell_voltage == pytest.approx(state_stoich.cell_voltage, rel=1e-8)
        assert state_flow.hfr == pytest.approx(state_stoich.hfr, rel=1e-8)
        assert state_flow.ca.cl.ionomer_water_content == pytest.approx(
            state_stoich.ca.cl.ionomer_water_content, rel=1e-8)
        assert state_flow.an.cl.ionomer_water_content == pytest.approx(
            state_stoich.an.cl.ionomer_water_content, rel=1e-8)
        assert state_flow.membrane.water_content == pytest.approx(
            state_stoich.membrane.water_content, rel=1e-8)
        assert state_flow.ca.ch.inlet_gas_flow_rate == pytest.approx(
            state_stoich.ca.ch.inlet_gas_flow_rate, rel=1e-8)
        assert state_flow.an.ch.inlet_gas_flow_rate == pytest.approx(
            state_stoich.an.ch.inlet_gas_flow_rate, rel=1e-8)

    def test_gas_composition_round_trips(self):
        """dry_o2/dry_h2/RH recovered from a GasFlowState should reproduce
        the composition it was built from, independent of the full solve."""
        gfs = GasFlowState(
            temperature=T_OP, pressure=P_OP,
            gas_species_molar_flow_rates=np.array([2.1e-3, 7.6e-3, 0., 1.0e-3]),
        )
        side_conditions = gfs.to_side_conditions()
        recomposed = mrpd.SideConditions(
            inlet_temperature=side_conditions.inlet_temperature,
            outlet_pressure=side_conditions.outlet_pressure,
            dry_o2_mole_fraction=side_conditions.dry_o2_mole_fraction,
            dry_h2_mole_fraction=side_conditions.dry_h2_mole_fraction,
            inlet_relative_humidity=side_conditions.inlet_relative_humidity,
        )
        assert side_conditions.stoichiometry == 0.
        assert recomposed.inlet_pressure == pytest.approx(P_OP)
        assert side_conditions.dry_o2_mole_fraction + side_conditions.dry_h2_mole_fraction \
            == pytest.approx(gfs.gas.X[0] / (1 - gfs.gas.X[3]) + gfs.gas.X[2] / (1 - gfs.gas.X[3]))


class TestSetGasFlowStatesExplicitSteadyState:
    """ExplicitSteadyStateModel.solve() should populate inlet/outlet
    GasFlowState objects consistent with Faraday's law and water stoichiometry."""

    def test_populated_after_solve(self):
        cell = build_default_cell()
        conditions = _conditions()
        model = ExplicitSteadyStateModel()
        state = model.set_initial_conditions(cell, conditions)
        state = model.solve(cell, conditions, state)

        for side_state in (state.ca, state.an):
            assert isinstance(side_state.inlet_gas_flow_state, GasFlowState)
            assert isinstance(side_state.outlet_gas_flow_state, GasFlowState)

    def test_reactant_consumption_matches_faraday(self):
        cell = build_default_cell()
        conditions = _conditions()
        model = ExplicitSteadyStateModel()
        state = model.set_initial_conditions(cell, conditions)
        state = model.solve(cell, conditions, state)

        o2_consumed = (
            state.ca.inlet_gas_flow_state.gas_species_molar_flow_rates[index_o2]
            - state.ca.outlet_gas_flow_state.gas_species_molar_flow_rates[index_o2]
        )
        h2_consumed = (
            state.an.inlet_gas_flow_state.gas_species_molar_flow_rates[index_h2]
            - state.an.outlet_gas_flow_state.gas_species_molar_flow_rates[index_h2]
        )
        assert o2_consumed == pytest.approx(I_OP / (4 * FARADAY_CONSTANT) * cell.area, rel=1e-8)
        assert h2_consumed == pytest.approx(I_OP / (2 * FARADAY_CONSTANT) * cell.area, rel=1e-8)

    def test_n2_is_inert(self):
        """N2 is not consumed or produced -- inlet and outlet N2 flow match."""
        cell = build_default_cell()
        conditions = _conditions()
        model = ExplicitSteadyStateModel()
        state = model.set_initial_conditions(cell, conditions)
        state = model.solve(cell, conditions, state)

        for side_state in (state.ca, state.an):
            assert side_state.outlet_gas_flow_state.gas_species_molar_flow_rates[index_n2] == pytest.approx(
                side_state.inlet_gas_flow_state.gas_species_molar_flow_rates[index_n2], rel=1e-8)

    def test_water_production_matches_h2_consumption(self):
        """2 H2 + O2 -> 2 H2O: total H2O gained (both sides, gas + liquid)
        equals total H2 consumed."""
        cell = build_default_cell()
        conditions = _conditions()
        model = ExplicitSteadyStateModel()
        state = model.set_initial_conditions(cell, conditions)
        state = model.solve(cell, conditions, state)

        h2_consumed = (
            state.an.inlet_gas_flow_state.gas_species_molar_flow_rates[index_h2]
            - state.an.outlet_gas_flow_state.gas_species_molar_flow_rates[index_h2]
        )
        water_produced = 0.
        for side_state in (state.ca, state.an):
            water_produced += (
                (side_state.outlet_gas_flow_state.gas_species_molar_flow_rates[index_h2ov]
                 - side_state.inlet_gas_flow_state.gas_species_molar_flow_rates[index_h2ov])
                + (side_state.outlet_gas_flow_state.liquid_molar_flow_rate
                   - side_state.inlet_gas_flow_state.liquid_molar_flow_rate)
            )
        assert water_produced == pytest.approx(h2_consumed, rel=1e-8)

    def test_liquid_flow_non_negative(self):
        cell = build_default_cell()
        conditions = _conditions()
        model = ExplicitSteadyStateModel()
        state = model.set_initial_conditions(cell, conditions)
        state = model.solve(cell, conditions, state)

        for side_state in (state.ca, state.an):
            assert side_state.inlet_gas_flow_state.liquid_molar_flow_rate >= 0.
            assert side_state.outlet_gas_flow_state.liquid_molar_flow_rate >= 0.

    def test_not_populated_for_vectorised_solve(self):
        """GasFlowState models one operating point; a polarization-curve-style
        vectorised solve should leave the fields unset rather than silently
        computing them for only the first point."""
        cell = build_default_cell()
        conditions = _conditions(i=np.linspace(1e3, 2e4, 5))
        model = ExplicitSteadyStateModel()
        state = model.set_initial_conditions(cell, conditions)
        state = model.solve(cell, conditions, state)

        assert state.ca.inlet_gas_flow_state is None
        assert state.ca.outlet_gas_flow_state is None
        assert state.an.inlet_gas_flow_state is None
        assert state.an.outlet_gas_flow_state is None


class TestSetGasFlowStatesTransient:
    """TransientModel should populate the same fields, via both f_transient
    (return_state=True) and evaluate()."""

    def test_populated_via_evaluate(self):
        cell = build_default_cell()
        conditions = _conditions()
        model = TransientModel(n_memb_mesh=5)
        _, x0 = model.set_initial_conditions(cell, conditions)
        state = model.evaluate(cell, conditions, np.array([0.]), x0.reshape(-1, 1))

        for side_state in (state.ca, state.an):
            assert isinstance(side_state.inlet_gas_flow_state, GasFlowState)
            assert isinstance(side_state.outlet_gas_flow_state, GasFlowState)

    def test_populated_via_f_transient_return_state(self):
        cell = build_default_cell()
        conditions = _conditions()
        model = TransientModel(n_memb_mesh=5)
        _, x0 = model.set_initial_conditions(cell, conditions)
        _, state = model.f_transient(0., x0, cell, conditions, return_state=True)

        for side_state in (state.ca, state.an):
            assert isinstance(side_state.inlet_gas_flow_state, GasFlowState)
            assert isinstance(side_state.outlet_gas_flow_state, GasFlowState)

    def test_reactant_consumption_matches_faraday(self):
        cell = build_default_cell()
        conditions = _conditions()
        model = TransientModel(n_memb_mesh=5)
        _, x0 = model.set_initial_conditions(cell, conditions)
        _, state = model.f_transient(0., x0, cell, conditions, return_state=True)

        o2_consumed = (
            state.ca.inlet_gas_flow_state.gas_species_molar_flow_rates[index_o2]
            - state.ca.outlet_gas_flow_state.gas_species_molar_flow_rates[index_o2]
        )
        assert o2_consumed == pytest.approx(I_OP / (4 * FARADAY_CONSTANT) * cell.area, rel=1e-8)
