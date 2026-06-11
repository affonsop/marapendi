"""
Regression tests for the vendored ``marapendi.legacy`` physics stack used by
``notebooks/durasys-data/fuel_cell_model_legacy.py`` (JES paper model).

Verifies that ``FuelCell.compute_ui_curve`` (explicit steady-state model)
reproduces a fixed polarization curve for the published ``initial_parameters``,
guarding against regressions when the legacy package is touched.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

import marapendi.legacy as mrpd
from marapendi.components.operating_conditions import OperatingConditions as NewOperatingConditions
from marapendi.legacy.gas_composition import index_o2, index_n2, index_h2, index_h2ov

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "durasys-data"
sys.path.insert(0, str(NOTEBOOK_DIR))

from fuel_cell_model_legacy import create_fuel_cell, initial_parameters  # noqa: E402


@pytest.fixture(scope="module")
def fuel_cell():
    return create_fuel_cell(initial_parameters, case_id=1)


@pytest.fixture(scope="module")
def operating_conditions():
    stack_temperature = np.array([353.15])
    cathode_conditions = mrpd.OperatingConditions(
        inlet_temperature=353.15,
        inlet_relative_humidity=0.7,
        outlet_pressure=1.5e5,
        dry_o2_mole_fraction=0.21,
        dry_h2_mole_fraction=0,
        stoichiometry=2.5,
    )
    anode_conditions = mrpd.OperatingConditions(
        inlet_temperature=353.15,
        inlet_relative_humidity=0.7,
        outlet_pressure=1.5e5,
        dry_o2_mole_fraction=0,
        dry_h2_mole_fraction=1,
        stoichiometry=1.5,
    )
    return stack_temperature, cathode_conditions, anode_conditions


# Reference polarization curve obtained from the current implementation at
# `initial_parameters`, used as a regression baseline.
CURRENT_DENSITIES = np.array([1000., 5000., 10000., 15000., 20000.])  # A/m2
REF_VOLTAGES = np.array([0.827953, 0.747140, 0.681857, 0.616790, 0.534550])
REF_HFR = np.array([6.40883257e-06, 6.72021687e-06, 7.16901492e-06, 7.86298945e-06, 9.16308146e-06])
TOL = 1e-4


def test_polarization_curve_matches_reference(fuel_cell, operating_conditions):
    stack_temperature, cathode_conditions, anode_conditions = operating_conditions
    voltages = fuel_cell.compute_ui_curve(
        CURRENT_DENSITIES, stack_temperature, cathode_conditions, anode_conditions
    )
    np.testing.assert_allclose(voltages, REF_VOLTAGES, rtol=TOL)


def test_voltage_monotonically_decreasing(fuel_cell, operating_conditions):
    stack_temperature, cathode_conditions, anode_conditions = operating_conditions
    voltages = fuel_cell.compute_ui_curve(
        CURRENT_DENSITIES, stack_temperature, cathode_conditions, anode_conditions
    )
    assert np.all(np.diff(voltages) < 0)


def test_compute_ui_curve_with_new_operating_conditions(fuel_cell, operating_conditions):
    """``OperatingConditions.from_components`` lets ``compute_ui_curve`` accept the
    new ``marapendi.components.operating_conditions.OperatingConditions``."""
    stack_temperature, cathode_conditions, anode_conditions = operating_conditions

    # Reference run with legacy-style conditions, also recording the resulting
    # inlet gas flow rates (mol/s) for each side.
    voltages_ref = fuel_cell.compute_ui_curve(
        CURRENT_DENSITIES, stack_temperature, cathode_conditions, anode_conditions
    )
    ca_flow = fuel_cell.ca.ch.inlet_gas_flow_rate
    an_flow = fuel_cell.an.ch.inlet_gas_flow_rate

    def _flows(dry_o2, dry_h2, rh, total_flow, temperature, pressure):
        h2ov_pressure = rh * mrpd.water_saturation_pressure(temperature)
        h2ov_fraction = h2ov_pressure / pressure
        h2ov = total_flow * h2ov_fraction
        dry_total = total_flow - h2ov
        flows = np.zeros((4,) + dry_total.shape)
        flows[index_o2, ...] = dry_total * dry_o2
        flows[index_h2, ...] = dry_total * dry_h2
        flows[index_n2, ...] = dry_total * (1 - dry_o2 - dry_h2)
        flows[index_h2ov, ...] = h2ov
        return flows

    new_cathode_conditions = NewOperatingConditions(
        temperature=cathode_conditions.inlet_temperature,
        backpressure=cathode_conditions.outlet_pressure,
        inlet_gas_molar_flow_rates=_flows(
            cathode_conditions.dry_o2_mole_fraction, cathode_conditions.dry_h2_mole_fraction,
            cathode_conditions.inlet_relative_humidity, ca_flow,
            cathode_conditions.inlet_temperature, cathode_conditions.outlet_pressure,
        ),
        inlet_h2ol_molar_flow_rate=0.,
    )
    new_anode_conditions = NewOperatingConditions(
        temperature=anode_conditions.inlet_temperature,
        backpressure=anode_conditions.outlet_pressure,
        inlet_gas_molar_flow_rates=_flows(
            anode_conditions.dry_o2_mole_fraction, anode_conditions.dry_h2_mole_fraction,
            anode_conditions.inlet_relative_humidity, an_flow,
            anode_conditions.inlet_temperature, anode_conditions.outlet_pressure,
        ),
        inlet_h2ol_molar_flow_rate=0.,
    )

    voltages_new = fuel_cell.compute_ui_curve(
        CURRENT_DENSITIES, stack_temperature, new_cathode_conditions, new_anode_conditions
    )
    np.testing.assert_allclose(voltages_new, voltages_ref, rtol=TOL)


def test_high_frequency_resistance_matches_reference(fuel_cell, operating_conditions):
    stack_temperature, cathode_conditions, anode_conditions = operating_conditions
    fuel_cell.compute_ui_curve(
        CURRENT_DENSITIES, stack_temperature, cathode_conditions, anode_conditions
    )
    state = fuel_cell.to_state()
    hfr = fuel_cell.voltage_model.high_frequency_resistance(state)
    np.testing.assert_allclose(hfr, REF_HFR, rtol=TOL)


def test_to_state_matches_attributes(fuel_cell, operating_conditions):
    """``to_state()`` reports the same values as the underlying attributes."""
    stack_temperature, cathode_conditions, anode_conditions = operating_conditions
    voltages = fuel_cell.compute_ui_curve(
        CURRENT_DENSITIES, stack_temperature, cathode_conditions, anode_conditions
    )

    state = fuel_cell.to_state()

    np.testing.assert_allclose(state.cell_voltage, voltages)
    np.testing.assert_allclose(state.cell_voltage, fuel_cell.cell_voltage)
    np.testing.assert_allclose(state.mea_temperature, fuel_cell.mea_temperature)
    np.testing.assert_allclose(state.ca.cl.temperature, fuel_cell.ca.cl.temperature)
    np.testing.assert_allclose(state.membrane.water_content, fuel_cell.membrane.water_content)
    np.testing.assert_allclose(
        state.membrane.proton_resistance,
        fuel_cell.membrane.proton_resistance(
            fuel_cell.membrane.temperature, water_saturation=fuel_cell.ca.cl.liquid_saturation
        ),
    )

    assert len(state.layers) == 4
    assert state.sides == (state.ca, state.an)


def test_voltage_model_matches_calculate_cell_voltage(fuel_cell, operating_conditions):
    """``VoltageModel.compute_cell_voltage`` reproduces ``FuelCell.calculate_cell_voltage``."""
    stack_temperature, cathode_conditions, anode_conditions = operating_conditions
    fuel_cell.compute_ui_curve(
        CURRENT_DENSITIES, stack_temperature, cathode_conditions, anode_conditions
    )

    voltage_model = mrpd.VoltageModel()
    state = fuel_cell.to_state()
    state = voltage_model.compute_cell_voltage(state)

    np.testing.assert_allclose(state.cell_voltage, fuel_cell.cell_voltage, rtol=TOL)
    np.testing.assert_allclose(state.eta_act, fuel_cell.voltage_model.activation_overpotential(state), rtol=TOL)
    np.testing.assert_allclose(state.eta_ohm, fuel_cell.voltage_model.ohmic_overpotential(state), rtol=TOL)
    np.testing.assert_allclose(state.E_rev, fuel_cell.voltage_model.reversible_cell_voltage(state), rtol=TOL)
    np.testing.assert_allclose(state.ca.cl.theta_catalyst, fuel_cell.ca.cl.theta_catalyst, rtol=TOL)
    np.testing.assert_allclose(
        state.membrane.proton_resistance + fuel_cell.electrical_resistance,
        fuel_cell.voltage_model.high_frequency_resistance(state),
        rtol=TOL,
    )
