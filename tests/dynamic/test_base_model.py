"""Tests for marapendi.models.model — BaseModel and CellBaseModel."""
# ruff: noqa: E501
import numpy as np
import pytest
import marapendi.dynamic as mrpd

T_OP = 353.15
P_OP = 1.5e5
RH   = 0.7

IC = dict(cell_temperature=T_OP, cell_pressure=P_OP,
          ca_rh=RH, an_rh=RH, ca_dry_o2=0.21, an_dry_h2=1.0)

_PHYSICS = dict(
    memb_model=mrpd.PFSAModel(),
    cl_model=mrpd.PtCCatalystLayerModel(),
    gas_diffusion_model=mrpd.PorousGasResistanceModel(),
    darcy_transport_model=mrpd.DarcyTransportModel(),
    voltage_model=mrpd.VoltageModel(),
)


def _make_cell():
    reaction = mrpd.ElectrochemicalReaction(
        reference_exchange_current_density=2.47e-8,
        activation_energy=67e6, reaction_order=0.54,
        reference_activity=1e5, reference_temperature=T_OP,
        number_of_electrons=2, charge_transfer_coeff=0.5,
    )
    cl_kw = dict(
        thickness=10e-6, bulk_density=2010., bulk_specific_heat_capacity=710.,
        bulk_thermal_conductivity=0.25, L_Pt=0.3e-2, wt_Pt=0.4, ic_ratio=0.7,
        ecsa=45e3, ionomer=mrpd.Nafion_N21X, r_C=25e-9, K_abs=1e-13,
        theta_contact=95, reaction=reaction,
    )
    gdl_kw = dict(
        thickness=160e-6, eps_p=0.72, bulk_density=440.,
        bulk_specific_heat_capacity=710., bulk_thermal_conductivity=1.24,
        K_abs=1e-12, theta_contact=115., tort=3,
    )
    return mrpd.Cell(
        area=25e-4, electrical_resistance=30e-7, thermal_resistance=2e-4,
        ca=mrpd.CellSide(
            cl=mrpd.PtCCatalystLayer(**cl_kw),
            gdl=mrpd.PorousLayer(**gdl_kw),
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        an=mrpd.CellSide(
            cl=mrpd.PtCCatalystLayer(**cl_kw),
            gdl=mrpd.PorousLayer(**gdl_kw),
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        memb=mrpd.Nafion_N212,
    )


def _make_base(current_density=5000.) -> mrpd.CellBaseModel:
    """Fresh fully-specified CellBaseModel."""
    return mrpd.CellBaseModel(
        transient_transport_model=mrpd.TransientCellModel(
            cell=_make_cell(), current_density=current_density,
        ),
        **_PHYSICS,
    )


# ─── module-scoped fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def base() -> mrpd.CellBaseModel:
    return _make_base()


# ─── BaseModel — generic composition mechanics ────────────────────────────────

class TestBaseModelInit:
    def test_n_states_equals_submodel(self, base):
        model = base.transient_transport_model
        assert base.n_states == model.n_states

    def test_missing_n_states_raises(self):
        class BadModel:
            pass
        with pytest.raises(AttributeError, match="n_states"):
            mrpd.BaseModel(submodels={'bad': BadModel()})

    def test_two_submodels_n_states_sum(self):
        # Compose two independent CellBaseModels — each exposes n_states.
        b_a = _make_base()
        b_b = _make_base()
        composed = mrpd.BaseModel(submodels={'a': b_a, 'b': b_b})
        assert composed.n_states == b_a.n_states + b_b.n_states

    def test_get_inputs_auto_registered(self, base):
        # BaseModel should have registered get_inputs from TransientCellModel
        assert 'transient_transport' in base.input_fns
        assert base.input_fns['transient_transport'](0.) == {'i': base.transient_transport_model.current_density}


# ─── CellBaseModel — initial state ───────────────────────────────────────────

class TestInitialState:
    def test_shape(self, base):
        y0 = base.initial_state(**IC)
        assert y0.shape == (base.transient_transport_model.n_states,)

    def test_all_finite(self, base):
        y0 = base.initial_state(**IC)
        assert np.all(np.isfinite(y0))

    def test_two_bases_concatenation(self):
        b_a = _make_base()
        b_b = _make_base()
        composed = mrpd.BaseModel(submodels={'a': b_a, 'b': b_b})
        y0_a = b_a.initial_state(**IC)
        y0_ab = composed.initial_state(a=IC, b=IC)
        n = b_a.n_states
        assert y0_ab.shape == (2 * n,)
        np.testing.assert_array_equal(y0_ab[:n], y0_a)
        np.testing.assert_array_equal(y0_ab[n:], y0_a)


# ─── CellBaseModel — rates_of_change ─────────────────────────────────────────

class TestRatesOfChange:
    def test_scalar_input(self, base):
        y0 = base.initial_state(**IC)
        dxdt = base.rates_of_change(0., y0)
        assert dxdt.shape == y0.shape
        assert np.all(np.isfinite(dxdt))

    def test_time_argument_ignored_for_constant_current(self, base):
        y0 = base.initial_state(**IC)
        d1 = base.rates_of_change(0.,   y0)
        d2 = base.rates_of_change(100., y0)
        np.testing.assert_array_almost_equal(d1, d2)

    def test_current_density_field_changes_rates(self):
        b_low  = _make_base(current_density=1000.)
        b_high = _make_base(current_density=10000.)
        y0_low  = b_low.initial_state(**IC)
        y0_high = b_high.initial_state(**IC)
        d_low  = b_low.rates_of_change(0., y0_low)
        d_high = b_high.rates_of_change(0., y0_high)
        assert not np.allclose(d_low, d_high)

    def test_callable_current_density(self):
        # Time-varying current: 0 A/m² for t < 50, 5000 A/m² after
        b = _make_base(current_density=lambda t: 0. if t < 50 else 5000.)
        y0 = b.initial_state(**IC)
        d_before = b.rates_of_change(0.,  y0)
        d_after  = b.rates_of_change(100., y0)
        assert not np.allclose(d_before, d_after)


# ─── BaseModel — split_state ──────────────────────────────────────────────────

class TestSplitState:
    def test_single_submodel(self, base):
        y0 = base.initial_state(**IC)
        parts = base.split_state(y0)
        assert 'transient_transport' in parts
        np.testing.assert_array_equal(parts['transient_transport'], y0)

    def test_two_submodels(self):
        b_a = _make_base()
        b_b = _make_base()
        composed = mrpd.BaseModel(submodels={'a': b_a, 'b': b_b})
        y0 = composed.initial_state(a=IC, b=IC)
        parts = composed.split_state(y0)
        assert parts['a'].shape == (b_a.n_states,)
        assert parts['b'].shape == (b_b.n_states,)
        np.testing.assert_array_equal(np.concatenate([parts['a'], parts['b']]), y0)

    def test_split_state_2d(self, base):
        y0 = base.initial_state(**IC)
        y_mat = np.stack([y0, y0], axis=1)
        parts = base.split_state(y_mat)
        assert parts['transient_transport'].shape == (base.transient_transport_model.n_states, 2)


# ─── BaseModel — solve_steady_state ──────────────────────────────────────────

class TestSolveSteadyState:
    @pytest.fixture(scope="class")
    def y0(self, base):
        return base.initial_state(**IC)

    @pytest.fixture(scope="class")
    def sol(self, base, y0):
        return base.solve_steady_state(y0, t=0.)

    def test_y_shape(self, base, sol):
        assert sol.y.shape == (base.n_states, 1)

    def test_y_finite(self, sol):
        assert np.all(np.isfinite(sol.y))

    def test_t_stored(self, sol):
        assert sol.t.shape == (1,)
        assert sol.t[0] == 0.

    def test_t_propagated(self, base, y0):
        sol = base.solve_steady_state(y0, t=42.5)
        assert sol.t[0] == 42.5

    def test_success_is_bool(self, sol):
        assert isinstance(sol.success, bool)

    def test_message_is_str(self, sol):
        assert isinstance(sol.message, str)

    def test_fun_shape(self, base, sol):
        assert sol.fun.shape == (base.n_states,)

    def test_residual_small_when_converged(self, sol):
        if sol.success:
            assert np.linalg.norm(sol.fun) < 1e-4

    def test_postprocess_compatible(self, base, sol):
        # sol.y has shape (n_states, 1), matching what postprocess expects
        i_density = base.transient_transport_model.current_density
        st = base.postprocess(sol.y, i_density=i_density)
        assert np.isfinite(float(st.V_cell[0]))

    def test_warm_start_improves_residual(self, base, y0):
        # Second solve from the converged solution should have a smaller residual
        sol1 = base.solve_steady_state(y0, t=0.)
        sol2 = base.solve_steady_state(sol1.y[:, 0], t=0.)
        assert np.linalg.norm(sol2.fun) <= np.linalg.norm(sol1.fun) + 1e-12
