import pytest
import numpy as np
import coulomb as cb
import cantera as ct

@pytest.fixture
def thick_membrane(): 
    return cb.Membrane(equivalent_weight=1100, density=1980, dry_thickness=125e-6)

@pytest.fixture
def thin_membrane(): 
    return cb.Membrane(equivalent_weight=1100, density=1980, dry_thickness=25e-6)

@pytest.fixture
def membrane_liso_2016(): 
    return cb.Membrane(equivalent_weight=1100, density=2000, dry_thickness=51e-6)   
 
@pytest.fixture
def fuel_cell_liso_2016(membrane_liso_2016): 
    fc = cb.FuelCell(cell_area=96e-4, cell_number=16, membrane=membrane_liso_2016)
    fc.membrane.temperature = 337.8842
    fc.current_density = np.linspace(0.25e4,1e4,4)
    fc.ca.cl.gas.set_temperature_and_pressure(337.8842, 135e3)
    fc.an.cl.gas.set_temperature_and_pressure(337.8842, 135e3)
    fc.ca.ch.gas.set_temperature_and_pressure(337.8842, 135e3)
    fc.ca.ch.gas.set_composition(0.2, 0, 1.)
    fc.an.ch.gas.set_temperature_and_pressure(337.8842, 135e3)
    fc.an.ch.gas.set_composition(0, 1.0, 0.2)
    fc.product_water_mass_source = fc.current_density * fc.cell_area * fc.cell_number / (2 * ct.faraday) * fc.ca.ch.gas.molecular_weights[-1]
    fc.ca.ch.o2_inlet_molar_flow_rate = fc.current_density * fc.cell_area * fc.cell_number / (4 * ct.faraday)
    fc.ca.ch.h2ov_inlet_molar_flow_rate = fc.ca.ch.o2_inlet_molar_flow_rate / fc.ca.ch.gas.X[0] * fc.ca.ch.gas.X[-1]
    fc.ca.ch.h2ov_inlet_mass_flow_rate = fc.ca.ch.h2ov_inlet_molar_flow_rate * fc.ca.ch.gas.molecular_weights[-1]
    return fc

@pytest.fixture 
def liso_2016_exp_data(): 
    return {
        0.2: np.array([0.0000365, 0.0000982, 0.0001487, 0.0001988]),
        0.8: np.array([0.0000500, 0.0001062, 0.0001464, 0.0002065])} 
    
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
    
    for pressure_difference, h2_crossover_flux in [(0, 0.0069e-6), (5 * 6895, 0.0085e-6)]:
        assert np.isclose(thick_membrane.hydrogen_permeation_flux(partial_pressure_h2 + pressure_difference, temperature, pressure_difference, water_vol_fraction), 
                          h2_crossover_flux, 0.1)

    for pressure_difference, h2_crossover_flux in [(0, 0.0229e-6), (5 * 6895, 0.0306e-6)]:
        assert np.isclose(thin_membrane.hydrogen_permeation_flux(partial_pressure_h2 + pressure_difference, temperature, pressure_difference, water_vol_fraction), 
                          h2_crossover_flux, 0.1)

import matplotlib.pyplot as plt 

def test_membrane_water_transport_model(fuel_cell_liso_2016, liso_2016_exp_data):
    fc = fuel_cell_liso_2016
    assert np.isclose(fc.ca.ch.gas.X[0], 0.1634, 1e-3)
    assert fc.ca.ch.gas.X[2] == 0
    fc.h2o_production = fc.current_density / (2 * ct.faraday)
    fc.membrane.water_balance_model.water_balance(fc)
    fc.ca.h2ov_outlet_mass_flow_rate = (fc.ca.ch.h2ov_inlet_mass_flow_rate + fc.product_water_mass_source -
                                        -fc.membrane.water_balance_model.cathode_flux(fc) * fc.cell_area * fc.cell_number) 
    plt.figure(figsize=(4,3))
    plt.plot(fc.current_density * 1e-4, 60e3*fc.ca.h2ov_outlet_mass_flow_rate, 'C0')
    plt.plot(fc.current_density * 1e-4, 60e3*liso_2016_exp_data[0.2], 'C0s', label='20 %')
    fc.an.ch.gas.set_composition(0, 1.0, 0.8)
    
    fc.membrane.water_balance_model.water_balance(fc)
    fc.ca.h2ov_outlet_mass_flow_rate = (fc.ca.ch.h2ov_inlet_mass_flow_rate + fc.product_water_mass_source -
                                        -fc.membrane.water_balance_model.cathode_flux(fc) * fc.cell_area * fc.cell_number)
    plt.plot(fc.current_density * 1e-4, 60e3*fc.ca.h2ov_outlet_mass_flow_rate, 'C1')
    plt.plot(fc.current_density * 1e-4, 60e3*liso_2016_exp_data[0.8], 'C1s', label='80 %')
    plt.ylim(0,0.35*60)
    plt.legend(title='RH$_{in,an}$')
    plt.xlabel('Current density (A/cm$^2$)')
    plt.ylabel('Cathode outlet water\nmass flow rate (g/min)')
    plt.tight_layout()
    plt.savefig('./tests/figures/test_membrane.png',dpi=300)
    for m_dot_h2o, m_dot_h2o_exp in zip(fc.ca.h2ov_outlet_mass_flow_rate, liso_2016_exp_data[0.2]): 
        assert np.isclose(m_dot_h2o, m_dot_h2o_exp, 0.5)
