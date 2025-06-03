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
            ionomer=mrpd.CatalystLayerIonomerModel(dry_density=2004, 
                                                   equivalent_weight=952, 
                                                   conductivity_correction=1., 
                                                   conductivity_exp=1.5),
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
            ionomer=mrpd.CatalystLayerIonomerModel(dry_density=2004, 
                                                   equivalent_weight=952, 
                                                   conductivity_correction=1., 
                                                   conductivity_exp=1.5),
        ),
        has_mpl=False, 
        has_gdl=False, 
    )    

@pytest.fixture 
def membrane():
    return mrpd.Membrane(equivalent_weight=1000/2.35, dry_density=1220, dry_thickness=60e-6)

@pytest.fixture
def electrolyzer_cell(cathode, anode, membrane): 
    return mrpd.ElectrolyzerCell(ca=cathode,an=anode, membrane=membrane, cell_area=5e-4, cell_number=1)

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
        inlet_liquid_saturation=1., 
        inlet_liquid=mrpd.KOH_solution(molarity=1),
        dry_h2_mole_fraction=0, 
        dry_o2_mole_fraction=0,
        outlet_pressure=1.5e5
    )

@pytest.fixture
def KOH_20_wt_percent(): 
    return mrpd.KOH_solution(temperature=298.15, 
                             weight_percent=20.37)
@pytest.fixture
def KOH_45_wt_percent(): 
    return mrpd.KOH_solution(temperature=298.15, 
                             weight_percent=45)

def test_electrochemistry():
    assert np.isclose(mrpd.electrochemistry.calculate_reversible_cell_voltage(298.15, 1e5, 1e5), 1.229, 1e-3)

def test_electrolyte(KOH_20_wt_percent, KOH_45_wt_percent): 
    assert np.isclose(KOH_20_wt_percent.density, 1190, rtol=1e-2)
    assert np.isclose(KOH_45_wt_percent.density, 1456, rtol=1e-2)
    assert np.isclose(KOH_45_wt_percent.molarity, 11.677, rtol=1e-2)
    assert np.isclose(KOH_45_wt_percent.ionic_conductivity, 0.4874, atol=0.01)
    assert np.isclose(KOH_20_wt_percent.ionic_conductivity, 0.5820, atol=0.01)
