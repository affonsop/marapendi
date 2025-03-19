import pytest
import numpy as np
import cantera as ct 

import coulomb as cb

@pytest.fixture
def fc(): 
    fc = cb.FuelCell(5e-4, 1.)
    fc.membrane.temperature = 353.15
    fc.current_density = 1.e4
    fc.ca.stoichiometry = 2.0
    fc.curent_density = 1e4
    fc.ca.ch = cb.GasFlowChannel(width=0.05e-2, height=0.08e-2, length=3.7e-2, n_parallel=14) # Values from Baker et al. (2009)
    fc.ca.ch.transport_resistance_model = cb.ChannelGasResistanceModel(A_ch=1.12, B_ch=1.01)
    fc.ca.ch.gas.set_temperature_and_pressure(353.15, 1.0e5)
    fc.ca.ch.gas.set_composition(0.02,0,0.62)
    fc.ca.ch.set_inlet_stoichiometry(2) 
    return fc

@pytest.fixture
def toray_gdl_060(): 
    lmbd = 0.86 # Data for figure 9 in Baker et al. (2009)
    f = 1 + 0.803 * np.exp(-1.17 * lmbd) + 0.197 * np.exp(-0.164 * lmbd)
    gdl = cb.PorousLayer(thickness=160e-6, 
                         gas=cb.GasComposition(temperature=353.15, pressure=1.0e5), 
                         effective_gas_diffusion_ratio=0.25/f) # D_OM / D_OMy = 4 in Baker et al. (2009)
    gdl.gas.set_temperature(353.15)
    gdl.gas.set_composition(0.2,0,0.62) 
    return gdl 

@pytest.fixture
def gas():
    gc = cb.GasComposition()
    gc.set_temperature_and_pressure(298.15,1e5)
    gc.set_composition(0.2, 0, 0)
    return gc

def test_gas_flow_rate(fc): 
    o2_molar_consumption = fc.current_density * fc.cell_number * fc.cell_area / (4 * ct.faraday)
    o2_molar_flow_rate = fc.ca.ch.inlet_stoichiometry * o2_molar_consumption 
    gas_molar_flow_rate = o2_molar_flow_rate / fc.ca.ch.o2_mole_fraction()
    gas_volume_flow_rate = gas_molar_flow_rate / fc.ca.ch.gas.concentration()
    assert  gas_volume_flow_rate == fc.ca.ch.calculate_inlet_gas_flow_rate(o2_molar_consumption)

def test_gas_diffusivity(gas):
    assert gas.X[0] == 0.2
    assert np.isclose(gas.species_diffusion_coefficient('o2'), 0.229e-4, 1e-2)

def test_gas_porous_transport_resistance(toray_gdl_060, fc): 
    o2_diffusion_coeff = fc.ca.ch.species_diffusion_coefficient('o2')
    non_dimensional_resistance = toray_gdl_060.gas_transport_resistance(species='o2') * o2_diffusion_coeff / fc.ca.ch.half_width
    assert np.isclose(non_dimensional_resistance, 3.73, 1e-2) # Dta from Baker et al. 2009, figure 9

def test_gas_flow_resistance(fc): 
    o2_diffusion_coeff = fc.ca.ch.species_diffusion_coefficient('o2')
    # Test for volume flow rate in Backer et al. (2009), stoichiometry higher than 10
    volume_flow_rate = 14 * 375e-6 / 60. / (1-fc.ca.ch.gas.X[-1]) * 1e5/ fc.ca.ch.gas.pressure * fc.ca.ch.gas.temperature / 273.15
    non_dim_channel_resistance = fc.ca.ch.gas_transport_resistance('o2', volume_flow_rate)  *  o2_diffusion_coeff / fc.ca.ch.half_width
    assert np.isclose(non_dim_channel_resistance, 1.73, 1e-2)
    # Test for stoichiometry 2 
    o2_molar_consumption = fc.current_density * fc.cell_number * fc.cell_area / (4 * ct.faraday)
    volume_flow_rate = fc.ca.ch.calculate_inlet_gas_flow_rate(o2_molar_consumption)
    non_dim_channel_resistance = fc.ca.ch.gas_transport_resistance('o2', volume_flow_rate)  *  o2_diffusion_coeff / fc.ca.ch.half_width
    assert np.isclose(non_dim_channel_resistance, 2.93, 1e-2)