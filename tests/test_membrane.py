"""Tests for marapendi.models.membrane — PFSAModel / MembraneModel."""
import numpy as np
import pytest
import marapendi as mrpd

T = 353.15  # K
MEMB = mrpd.Nafion_N212  # 50 µm Nafion


@pytest.fixture
def model():
    return mrpd.PFSAModel()


@pytest.fixture
def f_v(model):
    V_w = mrpd.water_molar_volume(T)
    return model.water_vol_fraction(10.0, V_w, MEMB.V_ion)


class TestWaterVolFraction:
    def test_zero_at_dry(self, model):
        V_w = mrpd.water_molar_volume(T)
        assert model.water_vol_fraction(0., V_w, MEMB.V_ion) == pytest.approx(0., abs=1e-12)

    def test_increases_with_lambda(self, model):
        V_w = mrpd.water_molar_volume(T)
        f5  = model.water_vol_fraction(5.,  V_w, MEMB.V_ion)
        f10 = model.water_vol_fraction(10., V_w, MEMB.V_ion)
        assert f10 > f5

    def test_bounded_below_one(self, model):
        V_w = mrpd.water_molar_volume(T)
        assert model.water_vol_fraction(20., V_w, MEMB.V_ion) < 1.0


class TestChargeCondutivity:
    def test_positive_for_proton(self, model, f_v):
        sigma = model.charge_conductivity(f_v, T, 'proton', MEMB)
        assert sigma > 0

    def test_zero_for_wrong_carrier(self, model, f_v):
        sigma = model.charge_conductivity(f_v, T, 'hydroxide', MEMB)
        assert sigma == pytest.approx(0., abs=1e-30)

    def test_increases_with_hydration(self, model):
        V_w = mrpd.water_molar_volume(T)
        f5  = model.water_vol_fraction(5.,  V_w, MEMB.V_ion)
        f10 = model.water_vol_fraction(10., V_w, MEMB.V_ion)
        s5  = model.charge_conductivity(f5,  T, 'proton', MEMB)
        s10 = model.charge_conductivity(f10, T, 'proton', MEMB)
        assert s10 > s5


class TestH2Permeability:
    def test_positive(self, model, f_v):
        assert model.h2_permeability(T, f_v) > 0

    def test_increases_with_hydration(self, model):
        V_w = mrpd.water_molar_volume(T)
        f_low  = model.water_vol_fraction(3.,  V_w, MEMB.V_ion)
        f_high = model.water_vol_fraction(12., V_w, MEMB.V_ion)
        assert model.h2_permeability(T, f_high) > model.h2_permeability(T, f_low)


class TestH2PermeationFlux:
    def test_positive(self, model, f_v):
        flux = model.calculate_h2_permeation_flux(T, f_v, 1e5, MEMB.thickness)
        assert flux > 0

    def test_proportional_to_pressure(self, model, f_v):
        f1 = model.calculate_h2_permeation_flux(T, f_v, 1e5, MEMB.thickness)
        f2 = model.calculate_h2_permeation_flux(T, f_v, 2e5, MEMB.thickness)
        assert f2 == pytest.approx(2 * f1, rel=1e-6)

    def test_inversely_proportional_to_thickness(self, model, f_v):
        f1 = model.calculate_h2_permeation_flux(T, f_v, 1e5, 50e-6)
        f2 = model.calculate_h2_permeation_flux(T, f_v, 1e5, 100e-6)
        assert f1 == pytest.approx(2 * f2, rel=1e-6)


class TestEquilibriumWaterContent:
    # sorption_coeffs_ion from a material is 1-D; polyval_vec needs 2-D rows.
    @pytest.fixture
    def coeffs_2d(self):
        return MEMB.sorption_coeffs_ion[np.newaxis, :]

    def test_increases_with_rh(self, model, coeffs_2d):
        rh = np.array([0.3, 0.5, 0.7, 0.9])
        lmbd = model.equilibrium_water_content(rh, coeffs_2d)
        assert np.all(np.diff(lmbd) > 0)

    def test_clipped_above_one(self, model, coeffs_2d):
        # rh > 1 should be clipped to 1 (pass as 1-D array, not scalar)
        lmbd_at_1  = model.equilibrium_water_content(np.array([1.0]), coeffs_2d)
        lmbd_above = model.equilibrium_water_content(np.array([1.5]), coeffs_2d)
        assert lmbd_at_1 == pytest.approx(lmbd_above, rel=1e-6)


class TestMembraneWaterResistance:
    # darken_num/den from a material are 1-D; the model always calls them via
    # Cell.build_property_arrays which produces 2-D (n_layers, n_coeffs) arrays.
    def _D_lmbd(self, model):
        num_2d = MEMB.darken_num_ion[np.newaxis, :]
        den_2d = MEMB.darken_den_ion[np.newaxis, :]
        return model.diffusion_coefficient(
            np.array([[10.]]), np.array([[T]]),
            num_2d, den_2d, MEMB.D_lmbd_ref_ion, MEMB.E_act_ion,
        )

    def test_positive_finite(self, model, f_v):
        D = self._D_lmbd(model)
        R = model.calculate_membrane_water_resistance(
            D, MEMB.thickness, MEMB.eps_ion, MEMB.c_ion, MEMB.tort_ion,
        )
        assert np.all(np.isfinite(R))
        assert np.all(R > 0)

    def test_proportional_to_thickness(self, model, f_v):
        D = self._D_lmbd(model)
        R1 = model.calculate_membrane_water_resistance(D, 50e-6,  MEMB.eps_ion, MEMB.c_ion, MEMB.tort_ion)
        R2 = model.calculate_membrane_water_resistance(D, 100e-6, MEMB.eps_ion, MEMB.c_ion, MEMB.tort_ion)
        assert R2 == pytest.approx(2 * R1, rel=1e-6)
