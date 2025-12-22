import pytest
import numpy as np
import marapendi as mrpd

@pytest.fixture
def flow_channel():
    # Mock Gas object with required attributes and methods
    gas = mrpd.GasComposition(300,101325)
    gas.set_composition(0.21,0, 0)

    # Create FlowChannel instance
    channel = mrpd.FlowChannel(
        height=1e-3,
        width=1e-3,
        length=10e-3,
        reactant='o2',
        n_parallel=2,
        gas=gas,
    )
    return channel

def test_flow_channel_initialization(flow_channel):
    assert flow_channel.height == 1e-3
    assert flow_channel.width == 1e-3
    assert flow_channel.length == 10e-3
    assert flow_channel.n_parallel == 2
    assert flow_channel.channel_flow_section == 1e-6  # width * height
    assert flow_channel.total_flow_section == 2e-6  # n_parallel * channel_flow_section
    assert flow_channel.hydraulic_diameter == pytest.approx(1e-3)  # 2wh/(w+h) for square channel

def test_set_inlet_stoichiometry(flow_channel):
    flow_channel.set_inlet_stoichiometry(2.0)
    assert flow_channel.inlet_stoichiometry == 2.0

def test_reactant_mole_fraction(flow_channel):
    assert flow_channel.reactant_mole_fraction() == 0.21  # X_o2

def test_set_fixed_inlet_gas_flow_rate(flow_channel):
    flow_channel.set_fixed_inlet_gas_flow_rate(1e-6)
    assert flow_channel.inlet_gas_flow_rate == 1e-6

def test_set_fixed_inlet_liquid_flow_rate(flow_channel):
    flow_channel.set_fixed_inlet_liquid_flow_rate(1e-7)
    assert flow_channel.inlet_liquid_flow_rate == 1e-7

def test_calculate_inlet_gas_flow_rate(flow_channel):
    flow_channel.set_inlet_stoichiometry(2.0)
    reactant_consumption = 1e-6  # kmol/s
    required_flow = flow_channel.calculate_inlet_gas_flow_rate(reactant_consumption)
    assert required_flow == 2.0 * 1e-6 / 0.21 / flow_channel.gas.concentration()

def test_gas_speed(flow_channel):
    # Inlet speed
    inlet_speed = flow_channel.gas_superficial_speed()
    assert inlet_speed == flow_channel.inlet_gas_flow_rate / flow_channel.total_flow_section

    # Outlet speed (mock outlet volumetric flow)
    outlet_volumetric_flow = 1.2e-6  # m³/s
    outlet_speed = flow_channel.gas_superficial_speed(outlet_volumetric_flow)
    assert outlet_speed == outlet_volumetric_flow / flow_channel.total_flow_section

def test_calculate_liquid_to_gas_velocity_ratio(flow_channel):
    # Mock the non_wetting_saturation attribute
    flow_channel.non_wetting_saturation = 0.3
    flow_channel.wetting_phase = 'gas'

    # Calculate the liquid-to-gas velocity ratio
    velocity_ratio = flow_channel.liquid_to_gas_velocity_ratio()

    # Expected liquid saturation
    liquid_saturation = flow_channel.non_wetting_saturation

    # Expected velocity ratio
    expected_ratio = (liquid_saturation / (1 - liquid_saturation)) ** 3 * (flow_channel.gas.mixture_kinematic_viscosity / mrpd.water.water_kinematic_viscosity(flow_channel.gas.temperature))

    # Assert the calculated ratio matches the expected ratio
    assert velocity_ratio == pytest.approx(expected_ratio)
