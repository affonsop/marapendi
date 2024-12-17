import coulomb as cb
import numpy as np

def test_water_saturation_pressure():
    assert np.isclose(cb.water_saturation_pressure(298.15),   3167, 1e-2)
    assert np.isclose(cb.water_saturation_pressure(353.15),  47414, 1e-2)
    assert np.isclose(cb.water_saturation_pressure(373.15), 101420, 1e-2)

def test_water_dew_point():
    assert np.isclose(cb.water_dew_point(3167),   298.15, 1e-2)
    assert np.isclose(cb.water_dew_point(47414),  353.15, 1e-2)
    assert np.isclose(cb.water_dew_point(101420), 373.15, 1e-2)

def test_water_dynamic_viscosity():
    assert np.isclose(cb.water_dynamic_viscosity(298.15),   8.9004e-4, 1e-3)
    assert np.isclose(cb.water_dynamic_viscosity(353.15),   3.5404e-4, 1e-3)

def test_water_saturation_concentration():
    assert np.isclose(cb.water_saturation_concentration(298.15), 3167 / (8.3145e3 * 298.15), 1e-2)

def test_water_density():
    assert np.isclose(cb.water_density(), 997., 1e-3)