import pytest
import numpy as np
import coulomb as cb 

def test_gas_composition(): 
    gas = cb.GasComposition(temperature=300, pressure=1e5)
    gas.set_composition(0.21, 0, 0.5)
    assert np.isclose(gas.gas.X[0], 0.21 * (1-0.5 * cb.water_saturation_pressure(gas.temperature())/1e5), 1e-4)
    assert gas.gas.X[1] == 0.79 * (1-0.5 * cb.water_saturation_pressure(gas.temperature())/1e5)
    assert gas.gas.X[2] == 0
    assert gas.gas.X[-1] == 0.5 * cb.water_saturation_pressure(gas.temperature())/1e5