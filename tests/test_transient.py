"""Tests for marapendi.models.transient — TransientCellModel."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
import marapendi as mrpd

cell_temperature = 353.15
cell_pressure    = 1.5e5
inlet_rh         = 0.7


# ─── shared fixture: fully assembled PEMFC ────────────────────────────────────

@pytest.fixture(scope="module")
def base():
    """Fully configured CellBaseModel."""
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
    return mrpd.CellBaseModel(
        transient_transport_model=mrpd.TransientCellModel(cell=cell, current_density=0.),
        memb_model=mrpd.PFSAModel(),
        cl_model=mrpd.PtCCatalystLayerModel(),
        gas_diffusion_model=mrpd.PorousGasResistanceModel(),
        darcy_transport_model=mrpd.DarcyTransportModel(),
        voltage_model=mrpd.VoltageModel(),
    )


@pytest.fixture(scope="module")
def model(base):
    return base.transient_transport_model


@pytest.fixture(scope="module")
def y0(base):
    return base.initial_state(
        cell_temperature=cell_temperature,
        cell_pressure=cell_pressure,
        ca_rh=inlet_rh,
        an_rh=inlet_rh,
        ca_dry_o2=0.21,
        an_dry_h2=1.0,
    )


# ─── layer-index integrity ────────────────────────────────────────────────────

class TestLayerIndices:
    def test_each_layer_has_unique_ix(self, model):
        indices = [layer.ix for layer in model.cell.layers]
        assert len(indices) == len(set(indices))

    def test_an_gdl_and_ca_gdl_distinct(self, model):
        assert model.cell.an.gdl.ix != model.cell.ca.gdl.ix

    def test_indices_cover_range(self, model):
        indices = sorted(layer.ix for layer in model.cell.layers)
        assert indices == list(range(model.n_layers))


# ─── initial state ────────────────────────────────────────────────────────────

class TestInitialState:
    def test_shape(self, model, y0):
        assert y0.shape == (model.n_layers * model.n_variables,)

    def test_all_finite(self, y0):
        assert np.all(np.isfinite(y0))

    def test_liquid_saturation_starts_at_zero(self, model, y0):
        x = y0.reshape(model.n_layers, model.n_variables) * model.norm_factor
        assert np.all(x[:, model.i_s] == pytest.approx(0.))

    def test_anode_gdl_has_nonzero_gas(self, model, y0):
        x = y0.reshape(model.n_layers, model.n_variables) * model.norm_factor
        gdl_ix = model.cell.an.gdl.ix
        assert np.all(x[gdl_ix, model.i_cg] >= 0)
        assert x[gdl_ix, model.i_cg[2]] > 0   # H2 present

    def test_cathode_has_oxygen_not_hydrogen(self, model, y0):
        x = y0.reshape(model.n_layers, model.n_variables) * model.norm_factor
        ca_ix = model.cell.ca.cl.ix
        assert x[ca_ix, model.i_cg[0]] > 0    # O2 present
        assert x[ca_ix, model.i_cg[2]] == pytest.approx(0.)  # no H2 by default

    def test_anode_has_hydrogen_not_oxygen(self, model, y0):
        x = y0.reshape(model.n_layers, model.n_variables) * model.norm_factor
        an_ix = model.cell.an.cl.ix
        assert x[an_ix, model.i_cg[2]] > 0    # H2 present
        assert x[an_ix, model.i_cg[0]] == pytest.approx(0.)  # no O2 by default


# ─── rates_of_change ──────────────────────────────────────────────────────────

class TestRatesOfChange:
    @pytest.mark.parametrize("i", [200., 2000., 10000.])
    def test_shape(self, model, y0, i):
        model.current_density = i
        dxdt = model.rates_of_change(y0[:, np.newaxis], i=i)
        assert dxdt.shape == (model.n_layers * model.n_variables, 1)

    @pytest.mark.parametrize("i", [200., 2000., 10000.])
    def test_all_finite(self, model, y0, i):
        model.current_density = i
        dxdt = model.rates_of_change(y0[:, np.newaxis], i=i)
        assert np.all(np.isfinite(dxdt)), f"NaN/inf in dxdt at i={i}"

    def test_channel_rates_are_zero(self, model, y0):
        model.current_density = 5000.
        dxdt = model.rates_of_change(y0[:, np.newaxis], i=5000.)[:, 0]
        n = model.n_variables
        for ch in (model.cell.an.ch, model.cell.ca.ch):
            row = dxdt[ch.ix * n: ch.ix * n + n]
            # T and gas concentrations frozen; lambda and s may or may not be
            assert row[model.i_T] == pytest.approx(0.)
            for ig in model.i_cg:
                assert row[ig] == pytest.approx(0.)
            assert row[model.i_s] == pytest.approx(0.)


# ─── short integration ────────────────────────────────────────────────────────

class TestShortIntegration:
    @pytest.fixture(scope="class")
    def sol(self, model, y0):
        return solve_ivp(
            fun=lambda _t, y: model.rates_of_change(y[:, np.newaxis], i=2000.)[:, 0],
            t_span=(0., 10.),
            y0=y0,
            method='BDF',
            max_step=5.,
            rtol=1e-3,
            atol=1e-6,
        )

    def test_solver_succeeds(self, sol):
        assert sol.status == 0, sol.message

    def test_no_nan_in_solution(self, sol):
        assert np.all(np.isfinite(sol.y))

    def test_saturation_stays_physical(self, model, sol):
        # Unpack s from last time step.  The BDF solver (atol=1e-6) can let s drift
        # slightly outside [0,1]; get_states_from_x clips it back during evaluation,
        # so we allow a small numerical margin here.
        x_end = (sol.y[:, -1].reshape(model.n_layers, model.n_variables)
                 * model.norm_factor)
        s_end = x_end[:, model.i_s]
        assert np.all(s_end >= -5e-3)
        assert np.all(s_end <= 1 + 1e-4)


# ─── voltage is physical ──────────────────────────────────────────────────────

class TestVoltage:
    def test_initial_voltage_below_reversible(self, model, y0):
        i = 5000.
        x = y0.reshape(model.n_layers, model.n_variables, 1) * model.norm_factor[..., np.newaxis]
        state = model._compute_derived_quantities(x, i)
        model._compute_voltage(state)
        E_rev = state.E_rev_ca - state.E_rev_an
        assert state.V_cell.item() < E_rev.item()

    def test_voltage_decreases_with_current(self, model, y0):
        voltages = []
        for i in [1000., 5000., 10000.]:
            x = y0.reshape(model.n_layers, model.n_variables, 1) * model.norm_factor[..., np.newaxis]
            state = model._compute_derived_quantities(x, i)
            model._compute_voltage(state)
            voltages.append(state.V_cell.item())
        assert voltages[0] > voltages[1] > voltages[2]
