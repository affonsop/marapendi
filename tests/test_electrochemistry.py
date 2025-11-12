import pytest
import numpy as np
import marapendi as mrpd

@pytest.fixture
def reaction_params():
    return mrpd.ElectrochemicalReaction(reference_exchange_current_density=0.0001, activation_energy=50e6,
                            reaction_order=1.5, reference_activity=0.8, reference_temperature=298)

def test_reversible_cell_voltage():
    assert  np.isclose(mrpd.calculate_reversible_cell_voltage(298.15, 1), 1.229, 1e-3)
    assert  np.isclose(mrpd.calculate_reversible_cell_voltage(353.15, .2**0.5 * 2), 1.1805, 1e-3)

def test_tafel_overpotential():
    assert np.isclose(mrpd.calculate_tafel_overpotential(1e4,1e-3,298.15,1,1.0), 0.41412, 1e-4)
    assert np.isclose(mrpd.calculate_tafel_overpotential(1e5,1e-3,298.15,1,1.0), 0.47328, 1e-4)
    assert np.isclose(mrpd.calculate_tafel_overpotential(1e3,1e-4,353.15,1,1.0), 0.49051, 1e-4)
    assert np.isclose(mrpd.calculate_tafel_overpotential(1e4,1e-3,353.15,1,1.0), 0.49051, 1e-4)
    assert np.isclose(mrpd.calculate_tafel_overpotential(1e4,1e-3,353.15,1,2.0), 0.24525, 1e-4)
    assert np.isclose(mrpd.calculate_tafel_overpotential(1e4,1e-3,353.15,1,1.0), 0.49051, 1e-4)
    
def test_exchange_current_density(reaction_params):
    assert np.isclose(mrpd.calculate_exchange_current_density(310, 0.5, reaction_params), 0.00010791, 1e-4)
    assert np.isclose(mrpd.calculate_exchange_current_density(310, 1.0, reaction_params), 0.00030522, 1e-4)
    assert np.isclose(mrpd.calculate_exchange_current_density(350, 1.0, reaction_params), 0.00280184, 1e-4)