import pytest
import numpy as np
import marapendi as mrpd
import matplotlib.pyplot as plt

@pytest.fixture 
def cathode(): 
    return mrpd.ElectrolyzerCellSide(
        cl=mrpd.PtCCatalystLayer(
            thickness=10e-6,
            ionomer_to_carbon_ratio=1./0.8,
            carbon_agglomerate_radius=25e-9,
            platinum_loading=0.5e-2, 
            catalyst_platinum_weight_percent=0.6,
            ionomer=mrpd.PAPIonomer(),
            contact_angle=95.,
            absolute_permeability=1e-12
        ),
        gdl=mrpd.PorousLayer(
            thickness=245e-6, 
            porosity=0.6, 
            absolute_permeability=1e-11,
            contact_angle=130., 
        ),
        has_mpl=False, 
        has_gdl=True, 
    )

@pytest.fixture 
def anode(): 
    return mrpd.ElectrolyzerCellSide(
        cl=mrpd.PorousTransferLayer(
            thickness=10e-6,
            porosity=0.83, 
            ionomer_to_catalyst_ratio=0.2,
            fiber_diameter=20e-6,
            catalyst_density=5400., 
            catalyst_loading=1e-2,
            ionomer=mrpd.PAPIonomer(),
            absolute_permeability=1e-12,
            contact_angle=60.,
        ),
        gdl=mrpd.PorousTransferLayer(
            thickness=450e-6,
            porosity=0.83, 
            ionomer_to_catalyst_ratio=0,
            fiber_diameter=20e-6,
            catalyst_density=5400., 
            catalyst_loading=0,
            absolute_permeability=1e-11,
            contact_angle=60.,
        ),
        has_mpl=False, 
        has_gdl=True, 
    )    

@pytest.fixture 
def membrane():
    return mrpd.PAP85(thickness=80e-6, 
                       water_balance_model=mrpd.MatrixMembraneWaterBalanceModel())

@pytest.fixture
def electrolyzer_cell(cathode, anode, membrane): 
    return mrpd.ElectrolyzerCell(ca=cathode,an=anode, membrane=membrane, area=5e-4, cell_number=1, electrical_resistance=60e-7)

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
        dry_o2_mole_fraction=1,
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
        inlet_gas_flow_rate=0,
    )
@pytest.fixture
def dry_cathode():
    return mrpd.OperatingConditions(
        inlet_temperature = 353.15,
        inlet_liquid_flow_rate=0.,
        inlet_relative_humidity=0., 
        inlet_liquid=mrpd.KOH_1M,
        dry_h2_mole_fraction=1, 
        dry_o2_mole_fraction=0,
        outlet_pressure=1.0e5, 
        inlet_gas_flow_rate=1e-12,
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
    assert np.isclose(mrpd.KOH_45_wt_percent.ionic_conductivity, 50, atol=1)
    assert np.isclose(mrpd.KOH_20_wt_percent.ionic_conductivity, 60, atol=1)
    assert np.isclose(mrpd.KOH_1M.molarity, 1, rtol=1e-3)
    assert np.isclose(mrpd.KOH_2M.molarity, 2, rtol=1e-3)
    assert np.isclose(mrpd.KOH_5M.molarity, 5, rtol=1e-3)
    assert np.isclose(mrpd.KOH_1M.solution_sat_pressure, 3054, rtol=1e-3)
    mrpd.KOH_20_wt_percent.set_temperature(293.15)
    mrpd.KOH_45_wt_percent.set_temperature(293.15)
    assert np.isclose(mrpd.KOH_20_wt_percent.surface_tension, 81.4e-3, rtol=1e-2)
    assert np.isclose(mrpd.KOH_45_wt_percent.surface_tension, 0.105, rtol=1e-2)

def test_operating_conditions(electrolyzer_cell, wet_anode, wet_cathode, dry_cathode, deionized_water_cathode, deionized_water_anode):     
    electrolyzer_cell.set_conditions(353.15, 1e4, wet_cathode, wet_anode)
    water_sat_pressure = mrpd.water_saturation_pressure(353.15)
    assert np.isclose(electrolyzer_cell.ca.ch.inlet_gas_flow_rate, 0)
    assert np.isclose(electrolyzer_cell.ca.ch.inlet_liquid_flow_rate, 200e-6/60.)
    assert np.isclose(electrolyzer_cell.temperature, 353.15)
    assert np.isclose(electrolyzer_cell.ca.electrolyte.solution_sat_pressure, 45616.95)
    assert np.isclose(electrolyzer_cell.ca.calculate_dry_gas_pressure(), 1e5 - electrolyzer_cell.ca.electrolyte.solution_sat_pressure)
    assert np.isclose(electrolyzer_cell.reversible_cell_voltage(), 1.4812)

    electrolyzer_cell.set_conditions(298.15, 0e4, dry_cathode, wet_anode)
    assert np.isclose(electrolyzer_cell.ca.cl.non_wetting_saturation, 0)
    assert np.isclose(dry_cathode.inlet_relative_humidity, 0)
    assert np.isclose(electrolyzer_cell.ca.cl.temperature, 298.15)
    assert np.isclose(electrolyzer_cell.ca.cl.relative_humidity(), 0)
    
    assert np.isclose(electrolyzer_cell.ca.cl.vapor_pressure(), electrolyzer_cell.ca.cl.relative_humidity() * electrolyzer_cell.ca.cl.saturation_pressure())
    assert np.isclose(electrolyzer_cell.ca.calculate_dry_gas_pressure(), 1e5 - electrolyzer_cell.ca.cl.vapor_pressure())
    assert np.isclose(electrolyzer_cell.an.calculate_dry_gas_pressure(), 1e5 - electrolyzer_cell.an.cl.vapor_pressure())
    assert np.isclose(electrolyzer_cell.reversible_cell_voltage(), 1.4812, rtol=1e-3)

    # electrolyzer_cell.set_conditions(353.15, 0e4, deionized_water_cathode, deionized_water_anode)
    # water_sat_pressure = mrpd.water_saturation_pressure(353.15)
    # assert np.isclose(electrolyzer_cell.ca.electrolyte.temperature, 353.15)
    # assert np.isclose(electrolyzer_cell.ca.electrolyte.solution_sat_pressure, water_sat_pressure)
    # assert np.isclose(electrolyzer_cell.ca.calculate_dry_gas_pressure(), 1e5 - water_sat_pressure)
    # assert np.isclose(electrolyzer_cell.reversible_cell_voltage(), 1.197)

def test_ohmic_resistance(electrolyzer_cell, deionized_water_anode, deionized_water_cathode, wet_anode, wet_cathode):
    electrolyzer_cell.set_conditions(353.15, 1e4, deionized_water_cathode, deionized_water_anode)
    electrolyzer_cell.ca.cl.ionomer_water_content = 20
    #assert np.isclose(electrolyzer_cell.ohmic_overpotential(), 1e-5)
    assert np.isclose(electrolyzer_cell.membrane.charge_resistance(20, electrolyzer_cell.temperature, 
                                               use_water_profile=False, charge='hydroxide'), 33.55e-7)
    
def test_water_balance(electrolyzer_cell, wet_anode, dry_cathode):
    i = np.array([0.1, 0.5, 1, 1.5, 2.]) * 1e4
    electrolyzer_cell.set_conditions(353.15, i, wet_anode, dry_cathode)
    electrolyzer_cell.calculate_water_transport()
    import time 
    t1 = time.time()
    electrolyzer_cell.explicit_steady_state_model()
    t2 = time.time()
    print((t2-t1)/len(i)*1e6)
    plt.figure(figsize=(4,3))
    plt.plot([1,2,3,4,5], electrolyzer_cell.ca.membrane_water_flux * 1e5, '-oC1')
    plt.bar([1,2,3,4,5],height=-electrolyzer_cell.ca.h2o_production * 1e5, bottom=0, width=.4, color='C0',alpha=0.4)
    plt.bar([1,2,3,4,5],
        height=(electrolyzer_cell.ca.membrane_water_flux+electrolyzer_cell.ca.h2o_production) * 1e5, 
        bottom= -electrolyzer_cell.ca.h2o_production * 1e5, width=.4)
    plt.ylim([0,40])
    plt.ylabel('Water crossover rate ($\mu$mol/cm$^2$.s)')
    plt.xlabel('Current density (A/cm$^2$)')
    plt.xlim([0.5,5.5])
    plt.xticks([1,2,3,4,5], labels=i*1e-4)
    plt.tight_layout()
    plt.savefig('./figures/test_aemwe_water_balance.png', bbox_inches='tight', dpi=300)
    plt.show()

    for side in (electrolyzer_cell.ca, electrolyzer_cell.an):
        np.testing.assert_allclose(
            side.h2o_production
            + side.membrane_water_flux,
            side.water_flux,
            rtol=1e-6,
            atol=0.0
        )

def test_polarization_curve_ss(electrolyzer_cell, wet_anode, dry_cathode):
    i = np.linspace(0.01e4,2e4,20)
    electrolyzer_cell.set_conditions(353.15, i, wet_anode, dry_cathode)
    electrolyzer_cell.calculate_water_transport()
    electrolyzer_cell.explicit_steady_state_model()
    plt.plot(i, electrolyzer_cell.cell_voltage, 's')
    plt.show()

# def test_polarization_curve_dynamic(electrolyzer_cell, wet_anode, dry_cathode):
#     i = lambda t: t * 2e4 / 3000
#     ec = electrolyzer_cell
#     ec.membrane.water_balance_model = mrpd.TransientMembraneWaterBalanceModel()

#     def f(t,x,u,p=None): 
#         ec.set_conditions(353.15, u['i'](t), wet_anode, dry_cathode)
#         ec.ca.cl.ionomer_water_content = x[0]
#         ec.membrane.water_content = x[1]
#         ec.an.cl.ionomer_water_content = x[2]
#         ec.ca.cl.liquid_saturation = x[3]
#         ec.an.cl.liquid_saturation = x[4]
#         ec.membrane.water_balance_model.solve_water_balance(electrolyzer_cell)
#         for side in (ec.ca, ec.an): 
#             water_flux = (
#                 side.membrane_water_flux + side.h2o_production
#             )

#         return dxdt
#     h = lambda t,x,u,p: x
#     mrpd.DynamicModel(f=f, h=h, u ={'i': i})

#     plt.plot(i, electrolyzer_cell.cell_voltage, 's')
#     plt.show()
