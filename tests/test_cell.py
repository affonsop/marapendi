"""
Tests for cell.py — Layer, CellSide, Cell, and Updatable.update_from_dict.
"""

import pytest
import numpy as np

import marapendi as mrpd
from marapendi.components.cell import CellSide, Cell
from marapendi.components.layer import Layer
from marapendi.components.membrane import Membrane
from marapendi.components.porous_layers import PorousLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_layer(**overrides):
    """Return a fully-specified Layer, optionally overriding individual fields."""
    defaults = dict(
        name="test_layer",
        thickness=1e-4,
        bulk_density=2000.0,
        porosity=0.4,
        tortuosity=1.5,
        pore_diameter=5e-6,
        absolute_permeability=1e-12,
        contact_angle=110.0,
        bulk_electrical_conductivity=1e4,
        bulk_specific_heat_capacity=800.0,
        bulk_thermal_conductivity=0.5,
        ionomer_vol_fraction=0.3,
    )
    defaults.update(overrides)
    return PorousLayer(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_cell():
    return Cell()


@pytest.fixture
def custom_cell():
    ca = CellSide(name="cathode")
    an = CellSide(name="anode")
    memb = Membrane()
    return Cell(name="custom", ca=ca, an=an, memb=memb)


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class TestLayer:
    def test_instantiation(self):
        layer = make_layer()
        assert layer.name == "test_layer"
        assert layer.thickness == pytest.approx(1e-4)
        assert layer.bulk_density == pytest.approx(2000.0)
        assert layer.porosity == pytest.approx(0.4)
        assert layer.tortuosity == pytest.approx(1.5)
        assert layer.pore_diameter == pytest.approx(5e-6)
        assert layer.absolute_permeability == pytest.approx(1e-12)
        assert layer.contact_angle == pytest.approx(110.0)
        assert layer.bulk_electrical_conductivity == pytest.approx(1e4)
        assert layer.bulk_specific_heat_capacity == pytest.approx(800.0)
        assert layer.bulk_thermal_conductivity == pytest.approx(0.5)
        assert layer.ionomer_vol_fraction == pytest.approx(0.3)

    def test_is_updatable(self):
        """Layer inherits from Updatable."""
        from marapendi.tools.tools import Updatable
        layer = make_layer(name="l")
        assert isinstance(layer, Updatable)


# ---------------------------------------------------------------------------
# CellSide
# ---------------------------------------------------------------------------

class TestCellSide:
    def test_default_instantiation(self):
        side = CellSide()
        assert side.name == "side"
        assert side.has_mpl is False
        assert side.has_gdl is True

    def test_layers_without_mpl(self):
        """Default: cl + gdl + ch → 3 layers in porous + ch."""
        side = CellSide(has_mpl=False, has_gdl=True)
        assert len(side.porous_layers) == 2  # cl + gdl
        assert len(side.layers) == 3         # cl + gdl + ch

    def test_layers_with_mpl(self):
        side = CellSide(has_mpl=True, has_gdl=True)
        assert len(side.porous_layers) == 3  # cl + mpl + gdl
        assert len(side.layers) == 4         # cl + mpl + gdl + ch

    def test_layers_without_gdl(self):
        side = CellSide(has_mpl=False, has_gdl=False)
        assert len(side.porous_layers) == 1  # cl only
        assert len(side.layers) == 2         # cl + ch

    def test_layers_with_mpl_without_gdl(self):
        side = CellSide(has_mpl=True, has_gdl=False)
        assert len(side.porous_layers) == 2  # cl + mpl
        assert len(side.layers) == 3         # cl + mpl + ch

    def test_ch_is_last_in_layers(self):
        side = CellSide()
        assert side.layers[-1] is side.ch


# ---------------------------------------------------------------------------
# Cell
# ---------------------------------------------------------------------------

class TestCell:
    def test_default_instantiation(self, default_cell):
        assert default_cell.name == "cell"
        assert isinstance(default_cell.ca, CellSide)
        assert isinstance(default_cell.an, CellSide)
        assert isinstance(default_cell.memb, Membrane)

    def test_layer_order(self, default_cell):
        """
        Expected order: an layers reversed + membrane + ca layers.
        With default CellSide (no MPL, has GDL): each side has [cl, gdl, ch].
        Reversed anode: [ch, gdl, cl]; then membrane; then [cl, gdl, ch] cathode.
        """
        c = default_cell
        expected_layers = c.an.layers[::-1] + [c.memb] + c.ca.layers
        assert c.layers == expected_layers

    def test_porous_layers_order(self, default_cell):
        c = default_cell
        expected = c.an.porous_layers[::-1] + c.ca.porous_layers
        assert c.porous_layers == expected

    def test_membrane_is_in_layers(self, default_cell):
        assert default_cell.memb in default_cell.layers

    def test_get_property_array_thickness(self, default_cell):
        arr = default_cell.get_property_array("thickness")
        assert isinstance(arr, np.ndarray)
        assert len(arr) == len(default_cell.layers)

    def test_get_property_array_bulk_thermal_conductivity(self, default_cell):
        arr = default_cell.get_property_array("bulk_thermal_conductivity")
        assert arr.shape == (len(default_cell.layers),)

    def test_build_property_arrays_keys(self, default_cell):
        """arrays dict should have keys matching fields of Layer."""
        from dataclasses import fields
        expected_keys = {f.name for f in fields(Layer)}
        assert expected_keys.issubset(set(default_cell.arrays.keys()))

    def test_build_property_arrays_values_are_arrays(self, default_cell):
        for key, val in default_cell.arrays.items():
            assert isinstance(val, np.ndarray), f"arrays['{key}'] is not an ndarray"

    def test_build_property_arrays_length(self, default_cell):
        n_layers = len(default_cell.layers)
        for key, val in default_cell.arrays.items():
            assert len(val) == n_layers, f"arrays['{key}'] has wrong length"

    def test_custom_name(self, custom_cell):
        assert custom_cell.name == "custom"


# ---------------------------------------------------------------------------
# Updatable.update_from_dict
# ---------------------------------------------------------------------------

class TestUpdatable:
    def test_scalar_update(self):
        cell = Cell()
        original_name = cell.name
        cell.update_from_dict({"name": "updated_cell"})
        assert cell.name == "updated_cell"
        assert cell.name != original_name

    def test_nested_dict_update(self):
        """Updating a nested dataclass field via dict."""
        cell = Cell()
        cell.update_from_dict({"memb": {"thickness": 50e-6}})
        assert cell.memb.thickness == pytest.approx(50e-6)

    def test_nested_side_update(self):
        cell = Cell()
        cell.update_from_dict({"ca": {"name": "new_cathode"}})
        assert cell.ca.name == "new_cathode"

    def test_invalid_nested_raises(self):
        """Passing a dict for a non-dataclass scalar attribute should raise TypeError."""
        cell = Cell()
        with pytest.raises(TypeError):
            cell.update_from_dict({"name": {"not": "valid"}})

    def test_cell_side_scalar_update(self):
        side = CellSide()
        side.update_from_dict({"name": "anode_updated"})
        assert side.name == "anode_updated"
