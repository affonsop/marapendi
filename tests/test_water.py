import marapendi as mrpd
import numpy as np
import pytest

@pytest.fixture 
def water(): 
    return mrpd.WaterProperties(300) 

def test_water_saturation_pressure():
    assert np.isclose(mrpd.water_saturation_pressure(298.15),   3167, 1e-2)
    assert np.isclose(mrpd.water_saturation_pressure(353.15),  47414, 1e-2)
    assert np.isclose(mrpd.water_saturation_pressure(373.15), 101420, 1e-2)

def test_water_dew_point():
    assert np.isclose(mrpd.water_dew_point(3167),   298.15, 1e-2)
    assert np.isclose(mrpd.water_dew_point(47414),  353.15, 1e-2)
    assert np.isclose(mrpd.water_dew_point(101420), 373.15, 1e-2)

def test_water_dynamic_viscosity():
    assert np.isclose(mrpd.water_dynamic_viscosity(298.15),   8.9004e-4, 1e-3)
    assert np.isclose(mrpd.water_dynamic_viscosity(353.15),   3.5404e-4, 1e-3)

def test_water_saturation_concentration():
    assert np.isclose(mrpd.water_saturation_concentration(298.15), 3167 / (8.3145e3 * 298.15), 1e-2)

def test_water_density():
    assert np.isclose(mrpd.water_density(), 997., 1e-2)

def test_water_molar_volume():
    assert np.isclose(mrpd.water_molar_volume(), 18.015 / 997., 1e-2)

def test_water_properties_class(water): 
    water.density == mrpd.water_density(300.)
    for temperature in [300., 353.15]:
        water.set_temperature(temperature)
        assert np.isclose(water.density, mrpd.water_density(temperature), 1e-2)
        assert np.isclose(water.molar_volume, mrpd.water_molar_volume(temperature), 1e-2)
        assert np.isclose(water.dynamic_viscosity, mrpd.water_dynamic_viscosity(temperature), 1e-2)
        assert np.isclose(water.saturation_pressure, mrpd.water_saturation_pressure(temperature), 1e-2)