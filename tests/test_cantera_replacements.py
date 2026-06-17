"""
Regression tests verifying that Cantera-free implementations agree with
previously computed Cantera reference values.

Tolerances reflect the stated accuracy of each correlation:
- h2_lhv / h2_hhv  : < 0.003 %  (polynomial fit over 300–473 K)
- water_dynamic_viscosity : < 1.1 %  (Vogel equation over 274–373 K)
- calculate_reversible_cell_voltage : < 0.01 %  (derived from hardcoded constants)
- GAS_CONSTANT / FARADAY_CONSTANT   : exact match to Cantera SI values
"""
import numpy as np
import pytest

from marapendi.models.electrochemistry import h2_lhv, h2_hhv, calculate_reversible_cell_voltage
from marapendi.models.water import water_dynamic_viscosity
from marapendi.models.constants import GAS_CONSTANT, FARADAY_CONSTANT


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_gas_constant_matches_cantera():
    """Cantera's ct.gas_constant == 8314.46261815324 J/(kmol·K)."""
    assert GAS_CONSTANT == pytest.approx(8314.46261815324, rel=1e-9)


def test_faraday_constant_matches_cantera():
    """Cantera's ct.faraday == 96485332.12331001 C/kmol."""
    assert FARADAY_CONSTANT == pytest.approx(96485332.12331001, rel=1e-9)


# ---------------------------------------------------------------------------
# h2_lhv — Cantera reference: h2ov.h(T) - 0.5*o2.h(T) - h2.h(T)
# ---------------------------------------------------------------------------

_lhv_cantera = [
    (300,    -241_843_016.3696),
    (320,    -242_042_972.8980),
    (353.15, -242_377_255.5101),
    (373.15, -242_579_537.9144),
    (400,    -242_850_583.8119),
    (450,    -243_350_127.8555),
    (473,    -243_576_372.8326),
]

@pytest.mark.parametrize("T,ref", _lhv_cantera)
def test_h2_lhv_vs_cantera(T, ref):
    """h2_lhv polynomial agrees with Cantera to within 0.003 %."""
    assert h2_lhv(T) == pytest.approx(ref, rel=3e-5)


def test_h2_lhv_vectorised():
    """h2_lhv broadcasts over arrays (no np.vectorize needed)."""
    T = np.array([300., 353.15, 473.])
    result = h2_lhv(T)
    assert result.shape == (3,)
    assert result[0] == pytest.approx(-241_843_016.37, rel=3e-5)


# ---------------------------------------------------------------------------
# h2_hhv — Cantera reference: h2ol.h(T) - 0.5*o2.h(T) - h2.h(T)
# ---------------------------------------------------------------------------

_hhv_cantera = [
    (300,    -285_769_568.6067),
    (320,    -285_140_015.9176),
    (353.15, -284_097_086.6349),
    (373.15, -283_457_410.3968),
    (400,    -282_582_487.5102),
    (450,    -280_905_194.9769),
    (473,    -280_105_381.1002),
]

@pytest.mark.parametrize("T,ref", _hhv_cantera)
def test_h2_hhv_vs_cantera(T, ref):
    """h2_hhv polynomial agrees with Cantera to within 0.003 %."""
    assert h2_hhv(T) == pytest.approx(ref, rel=3e-5)


def test_h2_hhv_vectorised():
    """h2_hhv broadcasts over arrays."""
    T = np.array([300., 353.15, 473.])
    result = h2_hhv(T)
    assert result.shape == (3,)
    assert result[0] == pytest.approx(-285_769_568.61, rel=3e-5)


# ---------------------------------------------------------------------------
# water_dynamic_viscosity — Cantera reference from ct.Water() at TQ=(T,0)
# ---------------------------------------------------------------------------

_viscosity_cantera = [
    (274,    1.741234e-03),
    (298.15, 8.904808e-04),
    (323,    5.484119e-04),
    (353.15, 3.543907e-04),
    (373,    2.822266e-04),
]

@pytest.mark.parametrize("T,ref", _viscosity_cantera)
def test_water_dynamic_viscosity_vs_cantera(T, ref):
    """Vogel equation agrees with Cantera to within 1.1 %."""
    assert water_dynamic_viscosity(T) == pytest.approx(ref, rel=1.1e-2)


# ---------------------------------------------------------------------------
# calculate_reversible_cell_voltage
# ---------------------------------------------------------------------------

def test_reversible_cell_voltage_standard_conditions():
    """At 298.15 K and unit activities the standard EMF is ~1.229 V."""
    v = calculate_reversible_cell_voltage(298.15, 1.0)
    assert v == pytest.approx(1.228870, rel=1e-4)


def test_reversible_cell_voltage_elevated_temperature():
    """Reversible voltage decreases with temperature (standard result)."""
    v_low  = calculate_reversible_cell_voltage(298.15, 1.0)
    v_high = calculate_reversible_cell_voltage(353.15, 1.0)
    assert v_high < v_low
