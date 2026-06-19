"""Integration tests for FuelCell steady-state simulation.

Uses a simplified cell (no MPL, default Nafion membrane) at 353 K to verify
the solve pipeline runs without error and produces physically plausible results.

All tests use the new model API:

    model = ExplicitSteadyStateModel()
    state = model.set_initial_conditions(cell, conditions)
    state = model.solve(cell, conditions, state)
"""
import numpy as np
import pytest
import marapendi as mrpd


def _make_cell():
    liq = mrpd.DarcyTransportModel(J_function_exponent=2)

    gdl = mrpd.GasDiffusionLayer(
        thickness=200e-6,
        porosity=0.6,
        contact_angle=120.,
        effective_gas_diffusion_ratio=0.3,
        absolute_permeability=1e-12,
        thermal_conductivity=0.5,
        two_phase_transport_model=liq,
    )

    ca_cl = mrpd.PtCCatalystLayer(
        ecsa=70e3,
        platinum_loading=0.4e-2,
        ionomer=mrpd.PFSAIonomer(),
        reaction=mrpd.ElectrochemicalReaction(
            reference_exchange_current_density=2.5e-4,
            reaction_order=0.54,
            activation_energy=67e6,
            reference_activity=1e5,
            reference_temperature=353.15,
            number_of_electrons=2,
            charge_transfer_coeff=0.5,
        ),
        thickness=10e-6,
        thermal_conductivity=0.22,
        pore_diameter=40e-9,
        absolute_permeability=1e-13,
        contact_angle=97.,
        two_phase_transport_model=liq,
    )

    return mrpd.FuelCell(
        area=25e-4,
        electrical_resistance=30e-7,
        ca=mrpd.FuelCellSide(
            cl=ca_cl,
            gdl=mrpd.GasDiffusionLayer(
                thickness=200e-6,
                effective_gas_diffusion_ratio=0.3,
                thermal_conductivity=0.5,
                two_phase_transport_model=liq,
            ),
            ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='o2'),
            has_mpl=False,
            thermal_contact_resistance=4e-4,
        ),
        an=mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(thickness=5e-6, two_phase_transport_model=liq),
            gdl=gdl,
            ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='h2'),
            has_mpl=False,
            thermal_contact_resistance=4e-4,
        ),
        membrane=mrpd.PFSA(
            ionomer=mrpd.PFSAIonomer(equivalent_weight=1100, dry_density=1980),
            dry_thickness=25e-6,
        ),
        use_eq_water_content_for_ionomer=True,
    )


T_OP = 353.15


def _conditions(i, T=T_OP, p=1.5e5, rh=0.5, st_ca=2.0, st_an=1.5):
    return mrpd.CellConditions(
        current_density=np.atleast_1d(i),
        cell_temperature=T,
        ca=mrpd.SideConditions(
            inlet_temperature=T, inlet_pressure=p, outlet_pressure=p,
            dry_o2_mole_fraction=0.21, inlet_relative_humidity=rh, stoichiometry=st_ca,
        ),
        an=mrpd.SideConditions(
            inlet_temperature=T, inlet_pressure=p, outlet_pressure=p,
            dry_h2_mole_fraction=1.0, inlet_relative_humidity=rh, stoichiometry=st_an,
        ),
    )


def _solve(cell, cond, model=None):
    if model is None:
        model = mrpd.ExplicitSteadyStateModel()
    state = model.set_initial_conditions(cell, cond)
    return model.solve(cell, cond, state)


@pytest.fixture
def cell():
    return _make_cell()


@pytest.fixture
def model():
    return mrpd.ExplicitSteadyStateModel()


class TestFuelCellConstruction:
    def test_ca_reactant(self, cell):
        assert cell.ca.reactant == 'o2'

    def test_an_reactant(self, cell):
        assert cell.an.reactant == 'h2'

    def test_porous_layers_list(self, cell):
        assert len(cell.ca.porous_layers) == 2

    def test_membrane_instantiated(self, cell):
        assert isinstance(cell.membrane, mrpd.PFSA)


class TestPolarizationCurve:
    def test_voltage_positive_at_low_current(self, cell, model):
        state = _solve(cell, _conditions(1e3), model)
        assert float(np.atleast_1d(state.cell_voltage)[0]) > 0.5

    def test_voltage_below_open_circuit(self, cell, model):
        state = _solve(cell, _conditions(1e3), model)
        assert float(np.atleast_1d(state.cell_voltage)[0]) < 1.23

    def test_voltage_decreases_with_current(self, cell, model):
        voltages = [
            float(np.atleast_1d(_solve(cell, _conditions(i), model).cell_voltage)[0])
            for i in [1e3, 5e3, 1e4]
        ]
        assert voltages[0] > voltages[1] > voltages[2]

    def test_state_has_mea_temperature(self, cell, model):
        state = _solve(cell, _conditions(5e3), model)
        assert state.mea_temperature is not None
        assert float(np.atleast_1d(state.mea_temperature)[0]) > T_OP

    def test_hfr_positive(self, cell, model):
        state = _solve(cell, _conditions(5e3), model)
        cell.state = state
        hfr = cell.high_frequency_resistance()
        assert float(np.atleast_1d(hfr)[0]) > 0

    def test_voltage_higher_at_elevated_pressure(self, cell, model):
        V_low  = float(np.atleast_1d(_solve(cell, _conditions(5e3, p=1.5e5), model).cell_voltage)[0])
        V_high = float(np.atleast_1d(_solve(cell, _conditions(5e3, p=2.5e5), model).cell_voltage)[0])
        assert V_high > V_low

    def test_vectorised_sweep(self, cell, model):
        i_arr = np.array([1e3, 5e3, 1e4, 2e4])
        state = _solve(cell, _conditions(i_arr), model)
        V = np.atleast_1d(state.cell_voltage)
        assert V.shape == i_arr.shape
        assert np.all(np.diff(V) < 0)

    def test_solve_returns_cell_state(self, cell, model):
        from marapendi.cell.state import CellState
        state = _solve(cell, _conditions(5e3), model)
        assert isinstance(state, CellState)
