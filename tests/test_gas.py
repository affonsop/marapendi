"""Tests for GasState and GasModel (models/gas.py)."""
import numpy as np
import pytest
import marapendi as mrpd
from marapendi.cell.state import LayerState, FlowChannelState


def _layer(temperature=353.15, pressure=1e5, o2=0.21, h2=0., rh=0.):
    state = LayerState(temperature=temperature, pressure=pressure)
    mrpd.GasModel.set_composition(state, o2, h2, rh, pressure, temperature)
    return state


def test_gas_state_default():
    gs = mrpd.GasState()
    assert gs.X[mrpd.species_indexes['o2']] == 1.
    assert gs.X[mrpd.species_indexes['h2o']] == 0.


def test_species_indexes_exhaustive():
    for species in ('o2', 'n2', 'h2', 'h2o'):
        assert species in mrpd.species_indexes
    assert len(mrpd.species_indexes) == 4


def test_set_composition_dry_air():
    state = _layer(o2=0.21, rh=0.)
    assert np.isclose(state.gas.X[mrpd.species_indexes['o2']], 0.21)
    assert np.isclose(state.gas.X[mrpd.species_indexes['n2']], 0.79)
    assert np.isclose(state.gas.X[mrpd.species_indexes['h2o']], 0.)


def test_set_composition_pure_hydrogen():
    state = _layer(o2=0., h2=1., rh=0.)
    assert np.isclose(state.gas.X[mrpd.species_indexes['h2']], 1.)
    assert np.isclose(state.gas.X[mrpd.species_indexes['o2']], 0.)


def test_set_composition_with_humidity():
    T, P, rh = 353.15, 1e5, 0.5
    state = _layer(T, P, o2=0.21, rh=rh)
    expected_h2o = rh * mrpd.water_saturation_pressure(T) / P
    assert np.isclose(state.gas.X[mrpd.species_indexes['h2o']], expected_h2o, rtol=1e-4)


def test_relative_humidity_round_trip():
    T, P, rh_in = 353.15, 1e5, 0.5
    state = _layer(T, P, rh=rh_in)
    assert np.isclose(mrpd.GasModel.relative_humidity(state), rh_in, rtol=1e-4)


def test_vapor_pressure():
    T, P, rh = 353.15, 1e5, 0.5
    state = _layer(T, P, rh=rh)
    expected = rh * mrpd.water_saturation_pressure(T)
    assert np.isclose(mrpd.GasModel.vapor_pressure(state), expected, rtol=1e-4)


def test_species_partial_pressure():
    T, P = 353.15, 2e5
    state = _layer(T, P, o2=0.21)
    pO2 = mrpd.GasModel.species_partial_pressure(state, 'o2')
    assert np.isclose(pO2, state.gas.X[mrpd.species_indexes['o2']] * P, rtol=1e-6)


def test_concentration_ideal_gas():
    T, P = 353.15, 1e5
    state = _layer(T, P)
    c = mrpd.GasModel.concentration(state)
    assert np.isclose(c, P / (mrpd.GAS_CONSTANT * T), rtol=1e-6)


def test_o2_diffusion_coefficient_order_of_magnitude():
    state = _layer(temperature=353.15, pressure=1e5, o2=0.21)
    D = mrpd.GasModel.species_diffusion_coefficient(state, 'o2')
    assert 1e-5 < D < 5e-5  # ~2.9e-5 at 353 K, 1 bar


def test_diffusion_scales_with_temperature():
    state_low = _layer(temperature=300., pressure=1e5)
    state_high = _layer(temperature=400., pressure=1e5)
    D_low = mrpd.GasModel.species_diffusion_coefficient(state_low, 'o2')
    D_high = mrpd.GasModel.species_diffusion_coefficient(state_high, 'o2')
    assert D_high > D_low


def test_diffusion_scales_inversely_with_pressure():
    state_low = _layer(temperature=353.15, pressure=1e5)
    state_high = _layer(temperature=353.15, pressure=3e5)
    D_low = mrpd.GasModel.species_diffusion_coefficient(state_low, 'o2')
    D_high = mrpd.GasModel.species_diffusion_coefficient(state_high, 'o2')
    assert D_low > D_high


def test_mixture_kinematic_viscosity_positive():
    state = _layer(temperature=353.15, pressure=1e5)
    nu = mrpd.GasModel.mixture_kinematic_viscosity(state)
    assert nu > 0


def test_saturation_pressure_cached():
    state = _layer(temperature=353.15, pressure=1e5)
    state.saturation_pressure = None
    p1 = mrpd.GasModel.saturation_pressure(state)
    p2 = mrpd.GasModel.saturation_pressure(state)
    assert p1 == p2
    assert np.isclose(p1, mrpd.water_saturation_pressure(353.15))


def test_h2o_diffusion_in_h2_atmosphere():
    # When H2 is present, H2-H2O diffusivity is used (higher than O2-H2O)
    state_h2 = _layer(o2=0., h2=1., rh=0.)
    state_air = _layer(o2=0.21, rh=0.)
    D_h2 = mrpd.GasModel.species_diffusion_coefficient(state_h2, 'h2o')
    D_air = mrpd.GasModel.species_diffusion_coefficient(state_air, 'h2o')
    assert D_h2 > D_air
