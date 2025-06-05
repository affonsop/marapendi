import pytest
import numpy as np
import marapendi as mrpd


@pytest.fixture 
def cathode(): 
    return mrpd.ElectrolyzerCellSide(
        cl=mrpd.PtCCatalystLayer(
            thickness=10e-6,
            ionomer_to_carbon_ratio=0.5,
            carbon_agglomerate_radius=25e-9,
            platinum_loading=0.3e-2, 
            catalyst_platinum_weight_percent=0.4,
            ionomer=mrpd.PAPIonomer(),
        ),
        gdl=mrpd.PorousLayer(
            thickness=245e-6, 
            porosity=0.79, 

        )
    )

@pytest.fixture 
def anode(): 
    return mrpd.ElectrolyzerCellSide(
        cl=mrpd.PorousTransferLayer(
            thickness=200e-6,
            porosity=0.83, 
            ionomer_to_catalyst_ratio=0.2,
            fiber_diameter=20e-6,
            catalyst_density=5400., 
            catalyst_loading=1e-2,
            ionomer=mrpd.PAPIonomer(),
        ),
        has_mpl=False, 
        has_gdl=False, 
    )    

@pytest.fixture 
def membrane():
    return mrpd.PAP85(dry_thickness=80e-6)

@pytest.fixture
def electrolyzer_cell(cathode, anode, membrane): 
    return mrpd.ElectrolyzerCell(ca=cathode,an=anode, membrane=membrane, cell_area=5e-4, cell_number=1, electrical_resistance=60e-7)

@pytest.fixture
def cathode_conditions():
    return mrpd.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_relative_humidity=.5,
        outlet_pressure=1.5e5, 
    )
@pytest.fixture
def anode_conditions():
    return mrpd.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_liquid_flow_rate=1., 
        inlet_liquid=mrpd.KOH_solution(molarity=1),
        dry_h2_mole_fraction=0, 
        dry_o2_mole_fraction=0,
        outlet_pressure=1.5e5
    )
@pytest.fixture
def wet_cathode():
    return mrpd.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_liquid_flow_rate=200e-6/60., # 200 mL/min
        inlet_liquid=mrpd.KOH_1M,
        dry_h2_mole_fraction=1, 
        dry_o2_mole_fraction=0,
        outlet_pressure=1.0e5, 
    )
@pytest.fixture
def wet_anode():
    return mrpd.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_liquid_flow_rate=200e-6/60., # 200 mL/min
        inlet_liquid=mrpd.KOH_1M,
        dry_h2_mole_fraction=0, 
        dry_o2_mole_fraction=1,
        outlet_pressure=1.0e5, 
    )
@pytest.fixture
def dry_cathode():
    return mrpd.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_liquid_flow_rate=0.,
        inlet_gas_flow_rate=200e-6/60., # 200 mL/min
        inlet_relative_humidity=0., 
        inlet_liquid=mrpd.KOH_1M,
        dry_h2_mole_fraction=1, 
        dry_o2_mole_fraction=0,
        outlet_pressure=1.0e5, 
    )

@pytest.fixture
def deionized_water_cathode():
    return mrpd.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_liquid_flow_rate=200e-6/60., # 200 mL/min
        inlet_liquid=mrpd.KOH_solution(temperature = 353.15, weight_percent=0., molality=0.),
        dry_h2_mole_fraction=1, 
        dry_o2_mole_fraction=0,
        outlet_pressure=1.0e5, 
    )
@pytest.fixture
def deionized_water_anode():
    return mrpd.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_liquid_flow_rate=200e-6/60., # 200 mL/min
        inlet_liquid=mrpd.KOH_solution(temperature = 353.15, weight_percent=0, molality=0.),
        dry_h2_mole_fraction=0, 
        dry_o2_mole_fraction=1,
        outlet_pressure=1.0e5, 
    )

def test_electrochemistry():
    assert np.isclose(mrpd.electrochemistry.calculate_reversible_cell_voltage(298.15, 1), 1.229, 1e-3)

def test_electrolyte(): 
    assert np.isclose(mrpd.KOH_20_wt_percent.density, 1190, rtol=1e-2)
    assert np.isclose(mrpd.KOH_45_wt_percent.density, 1456, rtol=1e-2)
    assert np.isclose(mrpd.KOH_45_wt_percent.molarity, 11.677, rtol=1e-2)
    assert np.isclose(mrpd.KOH_45_wt_percent.ionic_conductivity, 48.74, atol=1)
    assert np.isclose(mrpd.KOH_20_wt_percent.ionic_conductivity, 58.20, atol=1)
    assert np.isclose(mrpd.KOH_1M.molarity, 1, rtol=1e-4)
    assert np.isclose(mrpd.KOH_2M.molarity, 2, rtol=1e-4)
    assert np.isclose(mrpd.KOH_5M.molarity, 5, rtol=1e-4)
    assert np.isclose(mrpd.KOH_1M.solution_sat_pressure, 3054, rtol=1e-3)

def test_operating_conditions(electrolyzer_cell, wet_anode, wet_cathode, dry_cathode, deionized_water_cathode, deionized_water_anode):     
    electrolyzer_cell.set_conditions(353.15, 1e4, wet_cathode, wet_anode)
    water_sat_pressure = mrpd.water_saturation_pressure(353.15)
    assert np.isclose(electrolyzer_cell.ca.ch.inlet_gas_flow_rate, 0)
    assert np.isclose(electrolyzer_cell.ca.ch.inlet_liquid_flow_rate, 200e-6/60.)
    assert np.isclose(electrolyzer_cell.temperature, 353.15)
    assert np.isclose(electrolyzer_cell.ca.electrolyte.solution_sat_pressure, 45616.95)
    assert np.isclose(electrolyzer_cell.ca.calculate_dry_gas_pressure(), 1e5 - electrolyzer_cell.ca.electrolyte.solution_sat_pressure)
    assert np.isclose(electrolyzer_cell.reversible_cell_voltage(), 1.19564)

    electrolyzer_cell.set_conditions(298.15, 0e4, dry_cathode, dry_cathode)
    assert np.isclose(electrolyzer_cell.ca.cl.liquid_saturation, 0)
    assert np.isclose(dry_cathode.inlet_relative_humidity, 0)
    assert np.isclose(electrolyzer_cell.ca.cl.temperature, 298.15)
    assert np.isclose(electrolyzer_cell.ca.cl.relative_humidity(), 0)
    
    assert np.isclose(electrolyzer_cell.ca.cl.vapor_pressure(), electrolyzer_cell.ca.cl.relative_humidity() * electrolyzer_cell.ca.cl.saturation_pressure())
    assert np.isclose(electrolyzer_cell.ca.calculate_dry_gas_pressure(), 1e5 - electrolyzer_cell.ca.cl.vapor_pressure())
    assert np.isclose(electrolyzer_cell.an.calculate_dry_gas_pressure(), 1e5 - electrolyzer_cell.an.cl.vapor_pressure())
    assert np.isclose(electrolyzer_cell.reversible_cell_voltage(), 1.229, rtol=1e-3)

    electrolyzer_cell.set_conditions(353.15, 0e4, deionized_water_cathode, deionized_water_anode)
    water_sat_pressure = mrpd.water_saturation_pressure(353.15)
    assert np.isclose(electrolyzer_cell.ca.electrolyte.temperature, 353.15)
    assert np.isclose(electrolyzer_cell.ca.electrolyte.solution_sat_pressure, water_sat_pressure)
    assert np.isclose(electrolyzer_cell.ca.calculate_dry_gas_pressure(), 1e5 - water_sat_pressure)
    assert np.isclose(electrolyzer_cell.reversible_cell_voltage(), 1.197)

def test_ohmic_resistance(electrolyzer_cell, deionized_water_anode, deionized_water_cathode, wet_anode, wet_cathode):
    electrolyzer_cell.set_conditions(353.15, 1e4, deionized_water_cathode, deionized_water_anode)
    electrolyzer_cell.ca.cl.ionomer_water_content = 20
    #assert np.isclose(electrolyzer_cell.ohmic_overpotential(), 1e-5)
    assert np.isclose(electrolyzer_cell.membrane.charge_resistance(20, electrolyzer_cell.temperature, 
                                               use_water_profile=False, charge='hydroxide'), 53.8e-7)