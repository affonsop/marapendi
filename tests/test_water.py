"""Tests for water property correlations (models/water.py)."""
import numpy as np
import pytest
import marapendi as mrpd


def test_water_saturation_pressure():
    assert np.isclose(mrpd.water_saturation_pressure(298.15), 3167, rtol=1e-2)
    assert np.isclose(mrpd.water_saturation_pressure(353.15), 47414, rtol=1e-2)
    assert np.isclose(mrpd.water_saturation_pressure(373.15), 101420, rtol=1e-2)


def test_water_dew_point():
    assert np.isclose(mrpd.water_dew_point(3167), 298.15, rtol=1e-2)
    assert np.isclose(mrpd.water_dew_point(47414), 353.15, rtol=1e-2)
    assert np.isclose(mrpd.water_dew_point(101420), 373.15, rtol=1e-2)


def test_water_dynamic_viscosity():
    # Reference values validated against Cantera; Vogel equation accuracy ≤ 1.1 %
    assert np.isclose(mrpd.water_dynamic_viscosity(298.15), 8.9004e-4, rtol=1.5e-2)
    assert np.isclose(mrpd.water_dynamic_viscosity(353.15), 3.544e-4, rtol=1.5e-2)


def test_water_saturation_concentration():
    assert np.isclose(
        mrpd.water_saturation_concentration(298.15),
        3167 / (8.3145e3 * 298.15),
        rtol=1e-2,
    )


def test_water_density():
    assert np.isclose(mrpd.water_density(), 997., rtol=1e-2)


def test_water_molar_volume():
    assert np.isclose(mrpd.water_molar_volume(), 18.015 / 997., rtol=1e-2)


def test_water_surface_tension_range():
    # Roughly 0.07–0.075 N/m at 300 K
    sigma = mrpd.water_surface_tension(300.)
    assert 0.06 < sigma < 0.08


def test_water_kinematic_viscosity_consistent():
    mu = mrpd.water_dynamic_viscosity(298.15)
    rho = mrpd.water_density(298.15)
    nu = mrpd.water_kinematic_viscosity(298.15)
    assert np.isclose(nu, mu / rho, rtol=1e-6)
