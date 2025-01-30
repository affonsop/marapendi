import pytest
import numpy as np
import coulomb as cb 

@pytest.fixture 
def gas():
    return cb.GasComposition(temperature=300, pressure=1e5)


def test_gas_composition(gas): 
    gas.set_composition(0.21, 0, 0.5)
    assert np.isclose(gas.pressure, 1e5)
    assert np.isclose(gas.X[0], 0.21 * (1-0.5 * cb.water_saturation_pressure(gas.temperature)/1e5), 1e-4)
    assert np.isclose(gas.X[1], 0.79 * (1-0.5 * cb.water_saturation_pressure(gas.temperature)/1e5), 1e-4)
    assert gas.X[2] == 0
    assert np.isclose(gas.X[-1], 0.5 * cb.water_saturation_pressure(gas.temperature)/1e5, 1e-4)
    assert np.isclose(gas.pressure, 1e5)
    gas.set_pressure(2e5)
    assert np.isclose(gas.pressure, 2e5)
    assert np.isclose(gas.relative_humidity, 1)
    gas.set_temperature(353.15)
    assert np.isclose(gas.relative_humidity,
                      cb.water_saturation_pressure(300)/cb.water_saturation_pressure(353.15))

def test_vapor_pressure(gas): 
    gas.set_composition(0.21, 0, 0.5)
    assert np.isclose(gas.vapor_pressure(), 0.5 * cb.water_saturation_pressure(300))
