"""Tests for electrolyte models."""
import numpy as np
from marapendi.electrolyte.koh import KOH_solution


def test_koh_density():
    koh = KOH_solution(weight_percent=20)
    assert np.isclose(koh.density, 1190, rtol=2e-2)


def test_koh_density_increases_with_concentration():
    koh_low = KOH_solution(weight_percent=10)
    koh_high = KOH_solution(weight_percent=40)
    assert koh_high.density > koh_low.density


def test_koh_has_ionic_conductivity():
    koh = KOH_solution(weight_percent=20)
    assert koh.ionic_conductivity > 0
