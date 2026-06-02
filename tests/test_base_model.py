"""Tests for marapendi.models.model — BaseModel."""
import numpy as np
import pytest
import marapendi as mrpd

cell_temperature = 353.15
cell_pressure = 1.5e5
inlet_rh   = 0.7


def _make_model():
    reaction = mrpd.ElectrochemicalReaction(
        reference_exchange_current_density=2.47e-8,
        activation_energy=67e6,
        reaction_order=0.54,
        reference_activity=1e5,
        reference_temperature=cell_temperature,
        number_of_electrons=2,
        charge_transfer_coeff=0.5,
    )
    cl_kwargs = dict(
        thickness=10e-6, bulk_density=2010., bulk_specific_heat_capacity=710.,
        bulk_thermal_conductivity=0.25, L_Pt=0.3e-2, wt_Pt=0.4, ic_ratio=0.7,
        ecsa=45e3, ionomer=mrpd.Nafion_N21X, r_C=25e-9, K_abs=1e-13,
        theta_contact=95, reaction=reaction,
    )
    gdl_kwargs = dict(
        thickness=160e-6, eps_p=0.72, bulk_density=440.,
        bulk_specific_heat_capacity=710., bulk_thermal_conductivity=1.24,
        K_abs=1e-12, theta_contact=115., tort=3,
    )
    cell = mrpd.Cell(
        area=25e-4, electrical_resistance=30e-7, thermal_resistance=2e-4,
        memb_model=mrpd.PFSAModel(),
        cl_model=mrpd.PtCCatalystLayerModel(),
        ca=mrpd.CellSide(
            cl=mrpd.PtCCatalystLayer(**cl_kwargs),
            gdl=mrpd.PorousLayer(**gdl_kwargs),
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        an=mrpd.CellSide(
            cl=mrpd.PtCCatalystLayer(**cl_kwargs),
            gdl=mrpd.PorousLayer(**gdl_kwargs),
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        memb=mrpd.Nafion_N212,
    )
    return mrpd.TransientCellModel(cell=cell)


initial_conditions = dict(
    cell_temperature=cell_temperature, cell_pressure=cell_pressure,
    ca_rh=inlet_rh, an_rh=inlet_rh, ca_dry_o2=0.21, an_dry_h2=1.0,
)


@pytest.fixture(scope="module")
def single_model():
    return _make_model()


@pytest.fixture(scope="module")
def base(single_model):
    return mrpd.BaseModel(
        submodels={'cell': single_model},
        input_fns={'cell': lambda t: {'i': 5000.}},
    )


class TestBaseModelInit:
    def test_n_states_equals_submodel(self, base, single_model):
        assert base.n_states == single_model.n_states

    def test_missing_n_states_raises(self):
        class BadModel:
            pass
        with pytest.raises(AttributeError, match="n_states"):
            mrpd.BaseModel(submodels={'bad': BadModel()})

    def test_two_submodels_n_states_sum(self, single_model):
        m2 = mrpd.BaseModel(
            submodels={'a': single_model, 'b': single_model},
        )
        assert m2.n_states == 2 * single_model.n_states


class TestInitialState:
    def test_shape(self, base, single_model):
        y0 = base.initial_state(cell=initial_conditions)
        assert y0.shape == (single_model.n_states,)

    def test_all_finite(self, base):
        y0 = base.initial_state(cell=initial_conditions)
        assert np.all(np.isfinite(y0))

    def test_two_model_concatenation(self, single_model):
        base2 = mrpd.BaseModel(
            submodels={'a': single_model, 'b': single_model},
        )
        y0_a = single_model.initial_state(**initial_conditions)
        y0_ab = base2.initial_state(a=initial_conditions, b=initial_conditions)
        assert y0_ab.shape == (2 * single_model.n_states,)
        np.testing.assert_array_equal(y0_ab[:single_model.n_states], y0_a)
        np.testing.assert_array_equal(y0_ab[single_model.n_states:], y0_a)


class TestRatesOfChange:
    def test_scalar_input(self, base):
        y0 = base.initial_state(cell=initial_conditions)
        dxdt = base.rates_of_change(0., y0)
        assert dxdt.shape == y0.shape
        assert np.all(np.isfinite(dxdt))

    def test_time_argument_ignored_for_constant_input(self, base):
        y0 = base.initial_state(cell=initial_conditions)
        d1 = base.rates_of_change(0.,   y0)
        d2 = base.rates_of_change(100., y0)
        np.testing.assert_array_almost_equal(d1, d2)

    def test_time_dependent_input(self, single_model):
        # Input function switches current at t=50
        base_td = mrpd.BaseModel(
            submodels={'cell': single_model},
            input_fns={'cell': lambda t: {'i': 1000. if t < 50 else 10000.}},
        )
        y0 = base_td.initial_state(cell=initial_conditions)
        d_low  = base_td.rates_of_change(0.,  y0)
        d_high = base_td.rates_of_change(100., y0)
        # Rates should differ between the two current levels
        assert not np.allclose(d_low, d_high)


class TestSplitState:
    def test_single_submodel(self, base, single_model):
        y0 = base.initial_state(cell=initial_conditions)
        parts = base.split_state(y0)
        assert 'cell' in parts
        np.testing.assert_array_equal(parts['cell'], y0)

    def test_two_submodels(self, single_model):
        base2 = mrpd.BaseModel(
            submodels={'a': single_model, 'b': single_model},
        )
        y0 = base2.initial_state(a=initial_conditions, b=initial_conditions)
        parts = base2.split_state(y0)
        assert parts['a'].shape == (single_model.n_states,)
        assert parts['b'].shape == (single_model.n_states,)
        np.testing.assert_array_equal(
            np.concatenate([parts['a'], parts['b']]), y0
        )

    def test_split_state_2d(self, base, single_model):
        y0 = base.initial_state(cell=initial_conditions)
        y_mat = np.stack([y0, y0], axis=1)
        parts = base.split_state(y_mat)
        assert parts['cell'].shape == (single_model.n_states, 2)
