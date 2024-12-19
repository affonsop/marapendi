import pytest
import numpy as np
import coulomb as cb

@pytest.fixture
def thick_membrane(): 
    return cb.Membrane(equivalent_weight=1100, density=1980, thickness=125e-6)

@pytest.fixture
def thin_membrane(): 
    return cb.Membrane(equivalent_weight=1100, density=1980, thickness=25e-6)

@pytest.fixture
def water(): 
    return cb.WaterProperties(temperature=353.15)

def test_membrane_water_vol_fraction(thin_membrane, water): 
    assert np.isclose(thin_membrane.water_vol_fraction(10, water.molar_volume), 0.25, 1e-1)
    assert np.isclose(thin_membrane.water_vol_fraction(20, water.molar_volume), 0.42, 1e-1)

def test_membrane_hydrogen_permeability(thick_membrane, thin_membrane):
    # Values for hydrogen crossoover fluxes (mol/m2.s) were taken from Kang et al. (2021), figure 3.
    water_vol_fraction = 0.37 
    partial_pressure_h2 = 98100 # Pa
    temperature = 298.15 # K
    
    for pressure_difference, h2_crossover_flux in [(0, 0.0069e-3), (5 * 6895, 0.0085e-3)]:
        assert np.isclose(thick_membrane.hydrogen_permeation_flux(partial_pressure_h2 + pressure_difference, temperature, pressure_difference, water_vol_fraction), 
                          h2_crossover_flux, 0.1)

    for pressure_difference, h2_crossover_flux in [(0, 0.0229e-3), (5 * 6895, 0.0306e-3)]:
        assert np.isclose(thin_membrane.hydrogen_permeation_flux(partial_pressure_h2 + pressure_difference, temperature, pressure_difference, water_vol_fraction), 
                          h2_crossover_flux, 0.1)