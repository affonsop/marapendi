"""Tests for marapendi.models.voltage — VoltageModel."""
import numpy as np
import pytest
import cantera as ct
import marapendi.dynamic as mrpd

T = 353.15
MEMB = mrpd.Nafion_N212
STD_P = 1e5


@pytest.fixture
def voltage_model():
    return mrpd.VoltageModel()


@pytest.fixture
def memb_model():
    return mrpd.PFSAModel()


@pytest.fixture
def cl_model():
    return mrpd.PtCCatalystLayerModel()


@pytest.fixture
def ca_cl():
    return mrpd.PtCCatalystLayer(
        thickness=10e-6,
        bulk_density=2010.,
        bulk_specific_heat_capacity=710.,
        bulk_thermal_conductivity=0.25,
        L_Pt=0.3e-2,
        wt_Pt=0.4,
        ic_ratio=0.7,
        ecsa=45e3,
        ionomer=mrpd.Nafion_N21X,
        r_C=25e-9,
        K_abs=1e-13,
        theta_contact=95,
        reaction=mrpd.ElectrochemicalReaction(
            reference_exchange_current_density=2.47e-8,
            activation_energy=67e6,
            reaction_order=0.54,
            reference_activity=1e5,
            reference_temperature=353.15,
            number_of_electrons=2,
            charge_transfer_coeff=0.5,
        ),
    )


@pytest.fixture
def f_v(memb_model):
    V_w = mrpd.water_molar_volume(T)
    return memb_model.water_vol_fraction(10., V_w, MEMB.V_ion)


class TestReversibleVoltage:
    def test_above_one_volt(self, voltage_model):
        E_rev, _, _ = voltage_model.calculate_reversible_cell_voltage(
            T_an_cl=T, T_ca_cl=T, p_h2=STD_P, p_o2_local=STD_P * 0.21,
        )
        assert E_rev > 1.0

    def test_increases_with_o2_pressure(self, voltage_model):
        E1, _, _ = voltage_model.calculate_reversible_cell_voltage(T, T, STD_P, STD_P * 0.21)
        E2, _, _ = voltage_model.calculate_reversible_cell_voltage(T, T, STD_P, STD_P * 1.0)
        assert E2 > E1

    def test_increases_with_h2_pressure(self, voltage_model):
        E1, _, _ = voltage_model.calculate_reversible_cell_voltage(T, T, STD_P * 0.5, STD_P * 0.21)
        E2, _, _ = voltage_model.calculate_reversible_cell_voltage(T, T, STD_P * 1.5, STD_P * 0.21)
        assert E2 > E1

    def test_guard_against_zero_pressure(self, voltage_model):
        # Should not raise or produce NaN even at near-zero O2
        E_rev, _, _ = voltage_model.calculate_reversible_cell_voltage(T, T, STD_P, 1e-30)
        assert np.isfinite(E_rev)


class TestActivationOverpotential:
    def test_positive(self, voltage_model, ca_cl):
        eta = voltage_model.calculate_activation_overpotential(
            T_ca_cl=T, p_o2_local=STD_P * 0.21,
            i=5000., i_x=0., theta_PtO=0., ca_cl=ca_cl,
        )
        assert eta > 0

    def test_increases_with_current(self, voltage_model, ca_cl):
        eta1 = voltage_model.calculate_activation_overpotential(T, STD_P * 0.21, 1000., 0., 0., ca_cl)
        eta2 = voltage_model.calculate_activation_overpotential(T, STD_P * 0.21, 10000., 0., 0., ca_cl)
        assert eta2 > eta1


class TestOhmicOverpotential:
    def test_positive(self, voltage_model, memb_model, cl_model, ca_cl, f_v):
        eta_memb, eta_ca_cl, eta_gdl = voltage_model.calculate_ohmic_overpotential(
            T_an_cl=T, f_v_an_cl=f_v, T_memb=T, f_v_memb=f_v, T_ca_cl=T, f_v_ca_cl=f_v, s_ca_cl=0.,
            i=5000., memb=MEMB, electrical_resistance=30e-7,
            membrane_model=memb_model, ionomer_model=memb_model,
            ca_cl_model=cl_model, ca_cl=ca_cl, charge='proton',
        )
        assert eta_memb > 0
        assert eta_gdl > 0

    def test_proportional_to_current(self, voltage_model, memb_model, cl_model, ca_cl, f_v):
        kwargs = dict(T_an_cl=T, f_v_an_cl=f_v, T_memb=T, f_v_memb=f_v, T_ca_cl=T, f_v_ca_cl=f_v, s_ca_cl=0.,
                      memb=MEMB, electrical_resistance=30e-7,
                      membrane_model=memb_model, ionomer_model=memb_model,
                      ca_cl_model=cl_model, ca_cl=ca_cl, charge='proton')
        eta_memb1, _, eta_gdl1 = voltage_model.calculate_ohmic_overpotential(i=1000., **kwargs)
        eta_memb2, _, eta_gdl2 = voltage_model.calculate_ohmic_overpotential(i=2000., **kwargs)
        assert eta_memb2 == pytest.approx(2 * eta_memb1, rel=1e-6)
        assert eta_gdl2 == pytest.approx(2 * eta_gdl1, rel=1e-6)

    def test_simpson_uses_boundary_conditions(self, voltage_model, memb_model, cl_model, ca_cl, f_v):
        V_w = mrpd.water_molar_volume(T)
        f_v_low = memb_model.water_vol_fraction(5., V_w, MEMB.V_ion)
        base_kwargs = dict(T_memb=T, f_v_memb=f_v, T_ca_cl=T, f_v_ca_cl=f_v, s_ca_cl=0.,
                           i=5000., memb=MEMB, electrical_resistance=30e-7,
                           membrane_model=memb_model, ionomer_model=memb_model,
                           ca_cl_model=cl_model, ca_cl=ca_cl, charge='proton')
        eta_memb_same, _, _ = voltage_model.calculate_ohmic_overpotential(
            T_an_cl=T, f_v_an_cl=f_v, **base_kwargs
        )
        eta_memb_diff, _, _ = voltage_model.calculate_ohmic_overpotential(
            T_an_cl=T, f_v_an_cl=f_v_low, **base_kwargs
        )
        assert eta_memb_same != pytest.approx(eta_memb_diff)


class TestCellVoltage:
    """Tests for VoltageModel.compute_cell_voltage.

    A minimal CellState is constructed with p_o2_local pre-set so that the
    CL ionomer-film computation (compute_local_o2_partial_pressure) is
    bypassed — that is tested separately in test_catalyst_layer.py.
    """

    @pytest.fixture
    def fake_cell(self, memb_model, cl_model, ca_cl):
        """Minimal cell-like namespace accepted by compute_cell_voltage."""
        from types import SimpleNamespace
        ca_cl.electrolyte = mrpd.ElectrolyteSolution()
        return SimpleNamespace(
            memb=MEMB,
            electrical_resistance=30e-7,
            ca=SimpleNamespace(cl=ca_cl),
            charge='proton',
        )

    def _make_state(self, f_v, iF_val, p_o2_local=STD_P * 0.21):
        from marapendi.dynamic.components.cell_state import CellState
        from marapendi.dynamic.models.water import (
            water_saturation_concentration, water_density,
            water_kinematic_viscosity, water_molar_volume,
        )
        T_arr = np.array([[T]])
        fv = np.array([[f_v]])
        return CellState(
            x=np.zeros((1, 7, 1)), lmbd=np.zeros((1, 1)), T=T_arr,
            cg_k=np.zeros((1, 4, 1)), s=np.zeros((1, 1)),
            iF=np.array([[iF_val]]),
            c_sat=water_saturation_concentration(T_arr), c_v=np.zeros((1, 1)),
            rh=np.zeros((1, 1)), rho_l=water_density(T_arr),
            nu_l=water_kinematic_viscosity(T_arr), f_v=fv,
            T_memb=T_arr, T_ca_cl=T_arr, T_an_cl=T_arr,
            f_v_memb=fv, f_v_ca_cl=fv, f_v_an_cl=fv,
            lmbd_ca_cl=np.zeros((1, 1)),
            p_h2=np.array([[STD_P]]),
            p_o2_local=np.array([[p_o2_local]]),
        )

    def test_voltage_populated_on_state(self, voltage_model, f_v, fake_cell, memb_model, cl_model):
        state = self._make_state(f_v, 5000. / ct.faraday)
        voltage_model.compute_cell_voltage(state, fake_cell, memb_model, cl_model)
        assert state.V_cell is not None
        assert np.isfinite(state.V_cell).all()

    def test_voltage_below_reversible(self, voltage_model, f_v, fake_cell, memb_model, cl_model):
        state = self._make_state(f_v, 5000. / ct.faraday)
        voltage_model.compute_cell_voltage(state, fake_cell, memb_model, cl_model)
        E_rev = state.E_rev_ca - state.E_rev_an
        assert state.V_cell < E_rev

    def test_voltage_decreases_with_current(self, voltage_model, f_v, fake_cell, memb_model, cl_model):
        s1 = self._make_state(f_v, 1000. / ct.faraday)
        s2 = self._make_state(f_v, 10000. / ct.faraday)
        voltage_model.compute_cell_voltage(s1, fake_cell, memb_model, cl_model)
        voltage_model.compute_cell_voltage(s2, fake_cell, memb_model, cl_model)
        assert s2.V_cell < s1.V_cell
