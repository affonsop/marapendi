"""Tests for marapendi.models.catalyst_layer — PtCCatalystLayerModel."""
import numpy as np
import pytest
import marapendi.dynamic as mrpd

T = 353.15


@pytest.fixture
def cl():
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
def model():
    return mrpd.PtCCatalystLayerModel()


@pytest.fixture
def memb_model():
    return mrpd.PFSAModel()


class TestWaterFilmThickness:
    def test_zero_at_zero_saturation(self, model, cl):
        t = model.water_film_thickness(0., cl)
        assert t == pytest.approx(0., abs=1e-15)

    def test_positive_at_nonzero_saturation(self, model, cl):
        t = model.water_film_thickness(0.1, cl)
        assert t > 0

    def test_increases_with_saturation(self, model, cl):
        t1 = model.water_film_thickness(0.05, cl)
        t2 = model.water_film_thickness(0.20, cl)
        assert t2 > t1

    def test_vectorised(self, model, cl):
        s = np.linspace(0., 0.5, 6)
        t = model.water_film_thickness(s, cl)
        assert t.shape == s.shape
        assert np.all(np.diff(t) >= 0)


class TestO2IonomerFilmResistance:
    def test_positive_finite(self, model, cl, memb_model):
        t_ion  = cl.t_ion_film
        t_water = model.water_film_thickness(0.05, cl)
        R = model.o2_ionomer_film_resistance(10., T, cl, memb_model, t_ion, t_water)
        assert np.isfinite(R)
        assert R > 0

    def test_increases_with_water_film(self, model, cl, memb_model):
        t_ion = cl.t_ion_film
        t_thin = model.water_film_thickness(0.01, cl)
        t_thick = model.water_film_thickness(0.30, cl)
        R_thin  = model.o2_ionomer_film_resistance(10., T, cl, memb_model, t_ion, t_thin)
        R_thick = model.o2_ionomer_film_resistance(10., T, cl, memb_model, t_ion, t_thick)
        assert R_thick > R_thin


class TestEffectiveChargeResistance:
    # effective_charge_resistance calls electrolyte_sheet_resistance which
    # needs cl.electrolyte.  VoltageModel sets this as a workaround; mirror
    # that here so the test matches real usage.
    @pytest.fixture
    def cl_with_electrolyte(self, cl):
        cl.electrolyte = mrpd.ElectrolyteSolution()
        return cl

    def test_positive_finite(self, model, cl_with_electrolyte, memb_model):
        V_w = mrpd.water_molar_volume(T)
        f_v = memb_model.water_vol_fraction(10., V_w, cl_with_electrolyte.V_ion)
        R = model.effective_charge_resistance(
            i=5000., f_v=f_v, T=T, electrolyte_saturation=0.,
            charge='proton', ionomer_model=memb_model,
            cl=cl_with_electrolyte, reaction=cl_with_electrolyte.reaction,
        )
        assert np.isfinite(R)
        assert R > 0

    def test_neyerlin_correction_changes_result(self, model, cl_with_electrolyte, memb_model):
        V_w = mrpd.water_molar_volume(T)
        f_v = memb_model.water_vol_fraction(10., V_w, cl_with_electrolyte.V_ion)
        kwargs = dict(i=5000., f_v=f_v, T=T, electrolyte_saturation=0.,
                      charge='proton', ionomer_model=memb_model,
                      cl=cl_with_electrolyte, reaction=cl_with_electrolyte.reaction)
        R_no  = model.effective_charge_resistance(**kwargs, use_neyerlin_correction=False)
        R_yes = model.effective_charge_resistance(**kwargs, use_neyerlin_correction=True)
        assert R_no != pytest.approx(R_yes)

    def test_boundary_values_change_resistance(self, model, cl_with_electrolyte, memb_model):
        V_w = mrpd.water_molar_volume(T)
        f_v = memb_model.water_vol_fraction(10., V_w, cl_with_electrolyte.V_ion)
        kwargs = dict(i=5000., f_v=f_v, T=T, electrolyte_saturation=0.,
                      charge='proton', ionomer_model=memb_model,
                      cl=cl_with_electrolyte, reaction=cl_with_electrolyte.reaction)
        R_no_boundary = model.effective_charge_resistance(**kwargs)
        R_with_boundary = model.effective_charge_resistance(
            **kwargs, f_v_boundary=f_v * 1.5, T_boundary=T
        )
        assert R_no_boundary != pytest.approx(R_with_boundary)
