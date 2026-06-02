"""Tests for marapendi.models.water — water property functions."""
import numpy as np
import pytest
import marapendi as mrpd


T_COLD = 300.   # K
T_HOT  = 353.15  # K  (80 °C, typical PEMFC operating point)
T_BOIL = 373.15  # K


class TestWaterSaturationPressure:
    def test_increases_with_temperature(self):
        assert mrpd.water_saturation_pressure(T_HOT) > mrpd.water_saturation_pressure(T_COLD)

    def test_known_value_at_100c(self):
        # Antoine equation gives ~101325 Pa at 100 °C
        assert abs(mrpd.water_saturation_pressure(T_BOIL) - 101325) < 3000

    def test_vectorised(self):
        T = np.array([300., 320., 353.15])
        p = mrpd.water_saturation_pressure(T)
        assert p.shape == (3,)
        assert np.all(np.diff(p) > 0)


class TestWaterSaturationConcentration:
    def test_positive(self):
        assert mrpd.water_saturation_concentration(T_HOT) > 0

    def test_consistent_with_ideal_gas(self):
        import cantera as ct
        p_sat = mrpd.water_saturation_pressure(T_HOT)
        c_expected = p_sat / (ct.gas_constant * T_HOT)
        assert abs(mrpd.water_saturation_concentration(T_HOT) - c_expected) < 1e-6


class TestWaterDensity:
    def test_range(self):
        rho = mrpd.water_density(T_HOT)
        assert 900 < rho < 1000

    def test_decreases_with_temperature(self):
        assert mrpd.water_density(T_COLD) > mrpd.water_density(T_HOT)


class TestWaterViscosity:
    def test_kinematic_positive(self):
        assert mrpd.water_kinematic_viscosity(T_HOT) > 0

    def test_dynamic_positive(self):
        assert mrpd.water_dynamic_viscosity(T_HOT) > 0

    def test_kinematic_decreases_with_temperature(self):
        assert mrpd.water_kinematic_viscosity(T_HOT) < mrpd.water_kinematic_viscosity(T_COLD)


class TestWaterMolarVolume:
    def test_range(self):
        # ~18e-3 m³/kmol at standard conditions
        V = mrpd.water_molar_volume(T_HOT)
        assert 0.015 < V < 0.025


class TestWaterSurfaceTension:
    def test_positive(self):
        assert mrpd.water_surface_tension(T_HOT) > 0

    def test_decreases_with_temperature(self):
        assert mrpd.water_surface_tension(T_COLD) > mrpd.water_surface_tension(T_HOT)
