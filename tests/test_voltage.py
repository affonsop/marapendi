import pytest
import numpy as np
import coulomb as cb

conditions = {
    'p-out-ca': lambda t: 1.5e5, 
    'p-out-an': lambda t: 1.5e5, 
    'rh-in-ca': lambda t: 0.5, 
    'rh-in-an': lambda t: 0.5, 
    'st-ca'   : lambda t: 2.0,
    'st-an'   : lambda t: 1.2, 
    'i-dens'  : lambda t: 1e4, 
    'T-st'    : lambda t: 353.15,
}

params = {
    'exchange_current_density': 1e-8, 
    'ohmic_resistance': 25e-7, 
    'limit_current_density': 2, 
}



class FuelCell: 
    def __init__(self, params):
        self.exchange_current_density = params.get('exchange_current_density') 
        self.ohmic_resistance = params.get('ohmic_resistance')
        self.limit_current_density = params.get('limit_current_density')

    def cell_voltage(T,p_o2,p_h2):
        return cb.calculate_reversible_cell_voltage(T,p_o2,p_h2)
        
@pytest.fixture
def fuel_cell(): 
    return FuelCell(params)

@pytest.fixture
def operating_conditions():
    return OperatingConditions(conditions)

def test_reversible_cell_voltage():
    assert  np.isclose(cb.calculate_reversible_cell_voltage(temperature=298.15, partial_pressure_o2=1e5, partial_pressure_h2=1e5), 1.229, 1e-3)
    assert  np.isclose(cb.calculate_reversible_cell_voltage(temperature=353.15, partial_pressure_o2=.2e5, partial_pressure_h2=2e5), 1.256, 1e-3)

def test_tafel_overpotential():
    assert np.isclose(cb.calculate_tafel_overpotential(1e4,1e-3,298.15,2,0.5), 0.41412, 1e-4)
    assert np.isclose(cb.calculate_tafel_overpotential(1e5,1e-3,298.15,2,0.5), 0.47328, 1e-4)
    assert np.isclose(cb.calculate_tafel_overpotential(1e3,1e-4,353.15,2,0.5), 0.49051, 1e-4)
    assert np.isclose(cb.calculate_tafel_overpotential(1e4,1e-3,353.15,2,0.5), 0.49051, 1e-4)
    assert np.isclose(cb.calculate_tafel_overpotential(1e4,1e-3,353.15,2,1.0), 0.24525, 1e-4)
    assert np.isclose(cb.calculate_tafel_overpotential(1e4,1e-3,353.15,1,1.0), 0.49051, 1e-4)

def test_fuel_cell_voltage(fuel_cell, operating_conditions):
    assert fuel_cell.cell_voltage(operating_conditions) == 0 