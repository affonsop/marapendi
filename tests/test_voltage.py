from dataclasses import dataclass 

import pytest
import numpy as np
import coulomb as cb

@pytest.fixture
def orr_reaction_params():
    return cb.ElectrochemicalReaction(reference_exchange_current_density=2.47e-8 * 3e11 * 10e-6,
                                      activation_energy=67e6,
                                      reaction_order=0.54,
                                      reference_activity=1.,
                                      reference_temperature=353.15)

@pytest.fixture
def hor_reaction_params():
    return cb.ElectrochemicalReaction(reference_exchange_current_density=0.27 * 1e11 * 10e-6,
                                      activation_energy=16e6,
                                      reaction_order=0,
                                      reference_activity=1.,
                                      reference_temperature=353.15)


@dataclass 
class OperatingConditions: 
    temperature: float
    current_density: float
    partial_pressure_o2: float
    partial_pressure_h2: float


@pytest.fixture
def operating_conditions_1A_per_cm2():
    return OperatingConditions(
        temperature=353.15,
        current_density=1e4,
        partial_pressure_o2=0.3e5,
        partial_pressure_h2=1.5e5
    )

@pytest.fixture
def operating_conditions_ocv():
    return OperatingConditions(
        temperature=353.15,
        current_density=0.e4,
        partial_pressure_o2=0.3e5,
        partial_pressure_h2=1.5e5
    )

@pytest.fixture
def fuel_cell(orr_reaction_params, hor_reaction_params):
    fc = cb.FuelCell(
        cell_area=25e-4,
        cell_number=1,
        orr_reaction=orr_reaction_params,
        hor_reaction=hor_reaction_params,
    )
    fc.ca.cl.reaction = orr_reaction_params
    fc.an.cl.reaction = hor_reaction_params
    return fc

def test_fuel_cell_voltage_at_1Acm2(fuel_cell,operating_conditions_1A_per_cm2):
    assert np.isclose(fuel_cell.cell_voltage(operating_conditions_1A_per_cm2), 1.0701, 1e-4)

def test_fuel_cell_voltage_at_ocv(fuel_cell,operating_conditions_ocv):
    assert np.isclose(fuel_cell.cell_voltage(operating_conditions_ocv), 1.2602, 1e-4)