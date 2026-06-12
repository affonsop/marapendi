"""Tests for channel gas dynamics: FlowChannel additions, mu_g/mu_l fields,
TransientCellModel channel-dynamics path, and physical mass-balance checks."""
import numpy as np
import pytest
import cantera as ct

import marapendi.dynamic as mrpd
from marapendi.dynamic.models.water import water_saturation_pressure, water_molecular_weight

T = 353.15
P = 1.5e5


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def channel():
    return mrpd.FlowChannel(height=1e-3, width=1e-3, length=100e-3, n_parallel=14,
                             bulk_thermal_conductivity=100.)


@pytest.fixture
def conditions_pair():
    """InletAirConditions + InletHydrogenConditions at standard conditions."""
    p_sat = water_saturation_pressure(T)
    n_o2 = 1e-5   # kmol/s — arbitrary flow, only fractions matter for concentrations
    ca = mrpd.InletAirConditions(
        temperature=T, backpressure=P, rh_ref_pressure=P,
        o2_molar_flow_rate=n_o2, o2_dry_mole_fraction=0.21, inlet_rh=0.9,
    )
    an = mrpd.InletHydrogenConditions(
        temperature=T, backpressure=P, rh_ref_pressure=P,
        h2_molar_flow_rate=2 * n_o2, inlet_rh=0.9,
    )
    return ca, an


# ─── FlowChannel additions ────────────────────────────────────────────────────

class TestFlowChannelExtensions:
    def test_fRe_positive_and_finite(self, channel):
        # fRe is computed from aspect-ratio polynomial; square duct ≈ 14.2
        assert np.isfinite(channel.fRe) and channel.fRe > 0

    def test_fRe_square_duct(self):
        ch = mrpd.FlowChannel(height=1e-3, width=1e-3, bulk_thermal_conductivity=1.)
        assert ch.fRe == pytest.approx(14.23, rel=0.01)

    def test_total_volume(self, channel):
        expected = channel.total_flow_section * channel.length
        assert channel.total_volume == pytest.approx(expected)

    def test_total_volume_positive(self, channel):
        assert channel.total_volume > 0

    def test_superficial_velocity_matches_flow_section(self, channel):
        vol_flow = 1e-5   # m³/s
        assert channel.superficial_velocity(vol_flow) == pytest.approx(
            vol_flow / channel.total_flow_section
        )

    def test_superficial_velocity_proportional(self, channel):
        v1 = channel.superficial_velocity(1e-5)
        v2 = channel.superficial_velocity(2e-5)
        assert v2 == pytest.approx(2 * v1)


# ─── mu_g and mu_l on CellState ──────────────────────────────────────────────

def _make_base(conditions=None):
    nafion = mrpd.PFSAIonomer(
        rho_dry_ion=1.97e3, EW_ion=1020,
        darken_num_ion=np.array([0., 67.74, -32.03, 3.842]),
        darken_den_ion=np.array([103.37, -33.013, -2.115, 1.0]),
        sorption_coeffs_ion=np.array([0.043, 17.81, -39.85, 36.0]),
        lmbd_liq_ref_ion=22, D_lmbd_ref_ion=1e-10, k_des_ref_ion=1.42e-4,
        E_act_ion=20e6, E_act_cond_ion=15e6, sigma_ref_ion=116.,
        f_v_perc_ion=0.06, n_sigma_ion=1.5,
        T_ref_sigma_ion=353.15, T_ref_D_ion=353.15, T_ref_des_ion=353.15,
    )
    gdl = mrpd.PorousLayer(thickness=160e-6, eps_p=0.76, bulk_density=440.,
                            bulk_specific_heat_capacity=710., bulk_thermal_conductivity=1.6,
                            K_abs=6.15e-12, theta_contact=130., tort=1.6**2)
    orr = mrpd.ElectrochemicalReaction(
        reference_exchange_current_density=2.45e-4, activation_energy=67e6,
        reaction_order=0.54, reference_activity=1e5, reference_temperature=353.15,
        number_of_electrons=1, charge_transfer_coeff=1,
    )
    cell = mrpd.Cell(
        area=25e-4, electrical_resistance=1e-4, thermal_resistance=0,
        ca=mrpd.CellSide(
            cl=mrpd.PtCCatalystLayer(
                thickness=10e-6, bulk_density=1000., bulk_specific_heat_capacity=710.,
                bulk_thermal_conductivity=0.27, L_Pt=0.4e-2, wt_Pt=0.416, ic_ratio=1.04,
                ecsa=75e3, tort=1.6**2, ionomer=nafion, r_C=1e-10, K_abs=1e-13,
                theta_contact=95, reaction=orr,
            ),
            gdl=gdl, ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        an=mrpd.CellSide(
            cl=mrpd.PtCCatalystLayer(
                thickness=10e-6, bulk_density=1000., bulk_specific_heat_capacity=710.,
                bulk_thermal_conductivity=0.27, L_Pt=0.4e-2/3, wt_Pt=0.192, ic_ratio=1.07,
                ecsa=75e3, tort=1.6**2, ionomer=nafion, r_C=1e-10, K_abs=1e-13,
                theta_contact=95,
            ),
            gdl=gdl, ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        memb=mrpd.PFSAMembrane(thickness=25e-6, bulk_thermal_conductivity=0.3, ionomer=nafion),
    )
    tm_kwargs = dict(cell=cell, current_density=0.)
    if conditions is not None:
        tm_kwargs['conditions'] = conditions
    base = mrpd.CellBaseModel(
        transient_transport_model=mrpd.TransientCellModel(**tm_kwargs),
        memb_model=mrpd.PFSAModel(),
        cl_model=mrpd.PtCCatalystLayerModel(),
        gas_diffusion_model=mrpd.PorousGasResistanceModel(),
        darcy_transport_model=mrpd.DarcyTransportModel(),
        voltage_model=mrpd.VoltageModel(),
    )
    return base, cell


class TestViscosityFields:
    def test_mu_l_positive(self):
        base, cell = _make_base()
        y0 = base.initial_state(cell_temperature=T, cell_pressure=P,
                                  ca_rh=0.9, an_rh=0.9, ca_dry_o2=0.21, an_dry_h2=1.0)
        st = base.postprocess(y0[:, np.newaxis])
        assert st.mu_l is not None
        assert np.all(st.mu_l > 0)

    def test_mu_g_positive_in_gas_layers(self):
        # Membrane has no gas (cg=0) so mu_g=0 there; check only layers with gas.
        base, cell = _make_base()
        y0 = base.initial_state(cell_temperature=T, cell_pressure=P,
                                  ca_rh=0.9, an_rh=0.9, ca_dry_o2=0.21, an_dry_h2=1.0)
        st = base.postprocess(y0[:, np.newaxis])
        assert st.mu_g is not None
        gas_layers = [l for l in cell.layers if l is not cell.memb]
        gas_ix = [l.ix for l in gas_layers]
        assert np.all(st.mu_g[gas_ix] > 0)

    def test_mu_g_equals_nu_g_times_rho_g(self):
        base, cell = _make_base()
        y0 = base.initial_state(cell_temperature=T, cell_pressure=P,
                                  ca_rh=0.9, an_rh=0.9, ca_dry_o2=0.21, an_dry_h2=1.0)
        st = base.postprocess(y0[:, np.newaxis])
        np.testing.assert_allclose(st.mu_g, st.nu_g * st.rho_g, rtol=1e-10)

    def test_mu_l_equals_nu_l_times_rho_l(self):
        base, cell = _make_base()
        y0 = base.initial_state(cell_temperature=T, cell_pressure=P,
                                  ca_rh=0.9, an_rh=0.9, ca_dry_o2=0.21, an_dry_h2=1.0)
        st = base.postprocess(y0[:, np.newaxis])
        np.testing.assert_allclose(st.mu_l, st.nu_l * st.rho_l, rtol=1e-10)


# ─── Channel dynamics: frozen vs. active ─────────────────────────────────────

class TestChannelDynamicsFlag:
    """Verify that passing conditions= enables channel gas evolution."""

    @staticmethod
    def _make_conditions():
        n_o2 = 1e-5
        ca = mrpd.InletAirConditions(
            temperature=T, backpressure=P, rh_ref_pressure=P,
            o2_molar_flow_rate=n_o2, o2_dry_mole_fraction=0.21, inlet_rh=0.9,
        )
        an = mrpd.InletHydrogenConditions(
            temperature=T, backpressure=P, rh_ref_pressure=P,
            h2_molar_flow_rate=2 * n_o2, inlet_rh=0.9,
        )
        return mrpd.CellConditions(current_density=0., ca=ca, an=an)

    @staticmethod
    def _dxdt_2d(base, y0):
        """Zero-expand a compressed dxdt into (n_layers, n_variables)."""
        model = base.transient_transport_model
        dxdt_active = base.rates_of_change(0., y0)
        full = np.zeros(model.n_layers * model.n_variables)
        full[model._active_ix] = dxdt_active
        return full.reshape(model.n_layers, model.n_variables), model

    def test_no_conditions_freezes_channel_gas(self):
        base, cell = _make_base()
        y0 = base.initial_state(cell_temperature=T, cell_pressure=P,
                                  ca_rh=0.9, an_rh=0.9, ca_dry_o2=0.21, an_dry_h2=1.0)
        dxdt_2d, model = self._dxdt_2d(base, y0)
        for ch in (cell.ca.ch, cell.an.ch):
            np.testing.assert_array_equal(dxdt_2d[ch.ix, model.i_cg], 0.)

    def test_with_conditions_unfreezes_channel_gas(self):
        # Conditions must be passed at construction so BaseModel._slices is built
        # with the correct (larger) n_states; post-construction _rebuild_mask()
        # would change n_states but leave BaseModel._slices stale.
        cond = self._make_conditions()
        base, cell = _make_base(conditions=cond)
        y0 = base.initial_state(cell_temperature=T, cell_pressure=P,
                                  ca_rh=0.9, an_rh=0.9, ca_dry_o2=0.21, an_dry_h2=1.0)
        dxdt_2d, model = self._dxdt_2d(base, y0)
        for ch in (cell.ca.ch, cell.an.ch):
            assert not np.all(dxdt_2d[ch.ix, model.i_cg] == 0.)

    def test_channel_T_always_frozen(self):
        """Channel temperature dxdt is always zero regardless of conditions."""
        cond = self._make_conditions()
        base, cell = _make_base(conditions=cond)
        y0 = base.initial_state(cell_temperature=T, cell_pressure=P,
                                  ca_rh=0.9, an_rh=0.9, ca_dry_o2=0.21, an_dry_h2=1.0)
        dxdt_2d, model = self._dxdt_2d(base, y0)
        for ch in (cell.ca.ch, cell.an.ch):
            assert dxdt_2d[ch.ix, model.i_T] == pytest.approx(0.)


# ─── Physical: N₂ through-plane flux = 0 in porous layers ────────────────────

class TestN2ThroughPlaneFrozen:
    def test_n2_dxdt_zero_in_porous_layers(self):
        n_o2 = 1e-5
        ca = mrpd.InletAirConditions(
            temperature=T, backpressure=P, rh_ref_pressure=P,
            o2_molar_flow_rate=n_o2, o2_dry_mole_fraction=0.21, inlet_rh=0.9,
        )
        an = mrpd.InletHydrogenConditions(
            temperature=T, backpressure=P, rh_ref_pressure=P,
            h2_molar_flow_rate=2 * n_o2, inlet_rh=0.9,
        )
        cond = mrpd.CellConditions(current_density=0., ca=ca, an=an)
        base, cell = _make_base(conditions=cond)

        y0 = base.initial_state(cell_temperature=T, cell_pressure=P,
                                  ca_rh=0.9, an_rh=0.9, ca_dry_o2=0.21, an_dry_h2=1.0)
        model = base.transient_transport_model
        dxdt_active = base.rates_of_change(0., y0)
        full = np.zeros(model.n_layers * model.n_variables)
        full[model._active_ix] = dxdt_active
        dxdt_2d = full.reshape(model.n_layers, model.n_variables)
        i_N2 = model.i_cg[1]

        for layer in cell.layers:
            if layer not in (cell.ca.ch, cell.an.ch):
                assert dxdt_2d[layer.ix, i_N2] == pytest.approx(0., abs=1e-30), (
                    f"N2 dxdt != 0 in layer {layer} (ix={layer.ix})"
                )


# ─── Physical: gas mass balance at zero current ───────────────────────────────

class TestGasMassBalanceZeroCurrent:
    """At zero current with inlet=outlet (symmetric, high stoich), channel gas
    should be nearly at steady state — dxdt for reactive species should be
    small compared to their values.
    """

    def _modify_channel(self, model, y0, layer_ix, var_ix, value):
        """Expand active state, set y_full[layer_ix, var_ix] = value, compress back."""
        y_full = model.expand_state(y0).reshape(model.n_layers, model.n_variables)
        y_full[layer_ix, var_ix] = value
        return y_full.flatten()[model._active_ix]

    def _dxdt_at(self, base, y_active, layer_ix, var_ix):
        """Compute dxdt for one (layer, variable) entry via zero-expansion."""
        model = base.transient_transport_model
        dxdt_active = base.rates_of_change(0., y_active)
        full = np.zeros(model.n_layers * model.n_variables)
        full[model._active_ix] = dxdt_active
        return full.reshape(model.n_layers, model.n_variables)[layer_ix, var_ix]

    def test_empty_channel_fills_from_inlet(self):
        """With positive inlet O2 flow and zero channel O2, dxdt[ch, O2] > 0 (filling)."""
        n_o2 = 1e-5
        ca = mrpd.InletAirConditions(
            temperature=T, backpressure=P, rh_ref_pressure=P,
            o2_molar_flow_rate=n_o2, o2_dry_mole_fraction=0.21, inlet_rh=0.9,
        )
        an = mrpd.InletHydrogenConditions(
            temperature=T, backpressure=P, rh_ref_pressure=P,
            h2_molar_flow_rate=2 * n_o2, inlet_rh=0.9,
        )
        cond = mrpd.CellConditions(current_density=0., ca=ca, an=an)
        base, cell = _make_base(conditions=cond)
        model = base.transient_transport_model

        y0 = base.initial_state(cell_temperature=T, cell_pressure=P,
                                  ca_rh=0.9, an_rh=0.9, ca_dry_o2=0.21, an_dry_h2=1.0)
        # Zero cathode channel O2, then compress back to active state
        y0_mod = self._modify_channel(model, y0, cell.ca.ch.ix, model.i_cg[0], 0.)

        # Inlet brings in O2 → channel should fill (positive rate)
        assert self._dxdt_at(base, y0_mod, cell.ca.ch.ix, model.i_cg[0]) > 0

    def test_channel_drains_when_inlet_zero_and_pressure_above_backpressure(self):
        """Zero inlet flow but channel above backpressure: gas drains out (negative rate)."""
        ca = mrpd.InletAirConditions(
            temperature=T, backpressure=P, rh_ref_pressure=P,
            o2_molar_flow_rate=0.,
            o2_dry_mole_fraction=0.21, inlet_rh=0.9,
        )
        an = mrpd.InletHydrogenConditions(
            temperature=T, backpressure=P, rh_ref_pressure=P,
            h2_molar_flow_rate=0.,
            inlet_rh=0.9,
        )
        cond = mrpd.CellConditions(current_density=0., ca=ca, an=an)
        base, cell = _make_base(conditions=cond)
        model = base.transient_transport_model

        p_above = P * 1.01   # 1% above backpressure → gas drains
        y0 = base.initial_state(cell_temperature=T, cell_pressure=p_above,
                                  ca_rh=0.9, an_rh=0.9, ca_dry_o2=0.21, an_dry_h2=1.0)
        # No inlet, pressure above backpressure → gas flows out → negative rate
        assert self._dxdt_at(base, y0, cell.ca.ch.ix, model.i_cg[0]) < 0

    def test_n2_channel_dxdt_positive_with_inlet_n2(self):
        """N2 in channel should evolve (non-zero) since it has inlet/outlet flows."""
        n_o2 = 1e-5
        ca = mrpd.InletAirConditions(
            temperature=T, backpressure=P, rh_ref_pressure=P,
            o2_molar_flow_rate=n_o2, o2_dry_mole_fraction=0.21, inlet_rh=0.5,
        )
        an = mrpd.InletHydrogenConditions(
            temperature=T, backpressure=P, rh_ref_pressure=P,
            h2_molar_flow_rate=2 * n_o2, inlet_rh=0.5,
        )
        cond = mrpd.CellConditions(current_density=0., ca=ca, an=an)
        base, cell = _make_base(conditions=cond)
        model = base.transient_transport_model

        y0 = base.initial_state(cell_temperature=T, cell_pressure=P,
                                  ca_rh=0.5, an_rh=0.5, ca_dry_o2=0.21, an_dry_h2=1.0)
        # Zero cathode channel N2 to guarantee positive inlet-driven fill rate
        y0_mod = self._modify_channel(model, y0, cell.ca.ch.ix, model.i_cg[1], 0.)

        # With inlet N2 > 0 and channel N2 = 0, dxdt should be positive (filling)
        assert self._dxdt_at(base, y0_mod, cell.ca.ch.ix, model.i_cg[1]) > 0
