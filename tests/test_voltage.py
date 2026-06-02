"""Tests for marapendi.models.voltage — VoltageModel."""
import numpy as np
import pytest
import cantera as ct
import marapendi as mrpd

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
            T_memb=T, f_v_memb=f_v, T_ca_cl=T, f_v_ca_cl=f_v, s_ca_cl=0.,
            i=5000., memb=MEMB, electrical_resistance=30e-7,
            membrane_model=memb_model, ionomer_model=memb_model,
            ca_cl_model=cl_model, ca_cl=ca_cl, charge='proton',
        )
        assert eta_memb > 0
        assert eta_gdl > 0

    def test_proportional_to_current(self, voltage_model, memb_model, cl_model, ca_cl, f_v):
        kwargs = dict(T_memb=T, f_v_memb=f_v, T_ca_cl=T, f_v_ca_cl=f_v, s_ca_cl=0.,
                      memb=MEMB, electrical_resistance=30e-7,
                      membrane_model=memb_model, ionomer_model=memb_model,
                      ca_cl_model=cl_model, ca_cl=ca_cl, charge='proton')
        eta_memb1, _, eta_gdl1 = voltage_model.calculate_ohmic_overpotential(i=1000., **kwargs)
        eta_memb2, _, eta_gdl2 = voltage_model.calculate_ohmic_overpotential(i=2000., **kwargs)
        assert eta_memb2 == pytest.approx(2 * eta_memb1, rel=1e-6)
        assert eta_gdl2 == pytest.approx(2 * eta_gdl1, rel=1e-6)


class TestCellVoltage:
    @pytest.fixture
    def cell_voltage_kwargs(self, memb_model, cl_model, ca_cl, f_v):
        return dict(
            T_an_cl=T, T_ca_cl=T, T_memb=T,
            f_v_memb=f_v, f_v_ca_cl=f_v, s_ca_cl=0.,
            p_h2=STD_P, p_o2_local=STD_P * 0.21,
            i=5000., memb=MEMB, electrical_resistance=30e-7,
            memb_model=memb_model, ionomer_model=memb_model,
            ca_cl_model=cl_model, ca_cl=ca_cl, charge='proton',
        )

    def test_returns_tuple_of_eight(self, voltage_model, cell_voltage_kwargs):
        result = voltage_model.calculate_cell_voltage(**cell_voltage_kwargs)
        assert len(result) == 8

    def test_voltage_below_reversible(self, voltage_model, cell_voltage_kwargs):
        V_cell, _, _, E_rev_ca, E_rev_an, *_ = voltage_model.calculate_cell_voltage(
            **cell_voltage_kwargs
        )
        E_rev = E_rev_ca - E_rev_an
        assert V_cell < E_rev

    def test_voltage_decreases_with_current(self, voltage_model, cell_voltage_kwargs):
        V1, *_ = voltage_model.calculate_cell_voltage(**{**cell_voltage_kwargs, 'i': 1000.})
        V2, *_ = voltage_model.calculate_cell_voltage(**{**cell_voltage_kwargs, 'i': 10000.})
        assert V2 < V1
