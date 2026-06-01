import pytest
import numpy as np
import marapendi as mrpd
import cantera as ct

@pytest.fixture
def thick_membrane(): 
    return mrpd.PFSA(equivalent_weight=1100, bulk_density=1980, thickness=125e-6)

@pytest.fixture
def thin_membrane(): 
    return mrpd.PFSA(equivalent_weight=1100, bulk_density=1980, thickness=25e-6)

@pytest.fixture
def membrane_liso_2016(): 
    return mrpd.PFSA(equivalent_weight=1100., bulk_density=2000., thickness=51e-6, 
                     reference_absorption_coefficient=1.e-6, reference_water_diffusivity=5e-10,
                       water_balance_model=mrpd.MembraneWaterBalanceModel())   
 
@pytest.fixture
def fuel_cell_liso_2016(membrane_liso_2016): 
    fc = mrpd.FuelCell(area=96e-4, cell_number=16, membrane=membrane_liso_2016, 
                       ca=mrpd.FuelCellSide(cl=mrpd.PtCCatalystLayer(ionomer=mrpd.NafionD2020), 
                                            gdl=mrpd.PorousLayer(thickness=200e-6, effective_gas_diffusion_ratio=0.3, K_abs=1e-11, thermal_conductivity=.1)),
                       an=mrpd.FuelCellSide(cl=mrpd.PtCCatalystLayer(ionomer=mrpd.NafionD2020), 
                                            gdl=mrpd.PorousLayer(thickness=200e-6, effective_gas_diffusion_ratio=0.3, K_abs=1e-11,  thermal_conductivity=.1)),)
    fc.membrane.temperature = 337.8842
    fc.current_density = np.linspace(0.25e4,1e4,4)
    fc.ca.cl.gas.set_temperature_and_pressure(337.8842, 135e3)
    fc.ca.ch.gas.set_temperature_and_pressure(337.8842, 135e3)
    fc.ca.ch.gas.set_composition(0.21, 0, 1.)
    fc.product_water_mass_source = fc.current_density * fc.area * fc.cell_number / (2 * ct.faraday) * fc.ca.ch.gas.molecular_weights[-1]
    fc.ca.ch.o2_inlet_molar_flow_rate = fc.current_density * fc.area * fc.cell_number / (4 * ct.faraday)
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
    return mrpd.WaterProperties(temperature=353.15)

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
    plt.figure(figsize=(4,3))
    for k, rh_an in enumerate([0.2, 0.8]):
        fc.set_conditions(337.8842, np.array([0.25, 0.5, 0.75, 1]) * 1e4, 
            cathode_conditions = mrpd.OperatingConditions(
                inlet_temperature = 337.8842,
                inlet_relative_humidity = 1,
                outlet_pressure = 1.35e5,
                dry_o2_mole_fraction=0.21,
                dry_h2_mole_fraction=0,
                stoichiometry=2.4
            ),
            anode_conditions = mrpd.OperatingConditions(
                    inlet_temperature = 337.8842,
                    inlet_relative_humidity = rh_an,
                    outlet_pressure = 1.35e5,
                    dry_o2_mole_fraction=0,
                    dry_h2_mole_fraction=1,
                    stoichiometry=1.7
            )
        )
        
        fc.explicit_steady_state_model()
        
        fc.ca.h2ov_outlet_mass_flow_rate = (fc.ca.ch.h2ov_inlet_mass_flow_rate +
                                            fc.ca.water_flux * fc.cell_area * fc.cell_number * mrpd.water.water_molecular_weight) 
        print(fc.ca.water_flux, fc.an.water_flux, fc.ca.water_flux + fc.an.water_flux - fc.h2o_production)
        plt.plot(fc.current_density * 1e-4, 60e3*fc.ca.h2ov_outlet_mass_flow_rate, f'C{k}')
        plt.plot(fc.current_density * 1e-4, 60e3*liso_2016_exp_data[rh_an], f'C{k}s', label=f'{rh_an*100:.0f} %')

    plt.ylim(0,0.35*60)
    plt.legend(title='RH$_{in,an}$')
    plt.xlabel('Current density (A/cm$^2$)')
    plt.ylabel('Cathode outlet water\nmass flow rate (g/min)')
    plt.tight_layout()
    plt.savefig('./tests/figures/test_membrane.png',dpi=300)
    for m_dot_h2o, m_dot_h2o_exp in zip(fc.ca.h2ov_outlet_mass_flow_rate, liso_2016_exp_data[0.2]): 
        assert np.isclose(m_dot_h2o, m_dot_h2o_exp, atol=3e-5, rtol=0.05)


def test_equilibrium_water_content_basic(thin_membrane):
    rh, temperature = 0.5, 303.15
    result = thin_membrane.equilibrium_water_content(rh, temperature)
    expected = (0.043 + 17.18 * 0.5 - 39.85 * 0.5**2 + 36 * 0.5**3)
    assert np.isclose(result, expected)

def test_equilibrium_water_content_with_s_relax(thin_membrane):
    rh, temperature, s_relax = 0.5, 303.15, 0.5
    result = thin_membrane.equilibrium_water_content(rh, temperature, s_relax=0.5)
    phi = 0.15
    expected = (1-phi) * (0.043 + 17.18 * 0.5 - 39.85 * 0.5**2 + 36 * 0.5**3) + s_relax
    assert np.isclose(result, expected)
