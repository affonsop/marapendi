"""
Tests for cell.py — Layer, CellSide, Cell, and Updatable.update_from_dict.
"""

import pytest
import numpy as np

import marapendi.dynamic as mrpd
from marapendi.dynamic.components.cell import CellSide, Cell
from marapendi.dynamic.components.layer import Layer
from marapendi.dynamic.components.membrane import Membrane, PFSAMembrane
from marapendi.dynamic.components.porous_layers import PorousLayer
from marapendi.dynamic.components.ionomer import PFSAIonomer
from marapendi.dynamic.models.electrochemistry import ElectrochemicalReaction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_layer(**overrides):
    """Return a fully-specified Layer, optionally overriding individual fields."""
    defaults = dict(
        name="test_layer",
        thickness=1e-4,
        bulk_density=2000.0,
        eps_p=0.4,
        tort=1.5,
        d_p=5e-6,
        K_abs=1e-12,
        theta_contact=110.0,
        bulk_electrical_conductivity=1e4,
        bulk_specific_heat_capacity=800.0,
        bulk_thermal_conductivity=0.5,
        eps_ion=0.3,
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
        assert layer.eps_p == pytest.approx(0.4)
        assert layer.tort == pytest.approx(1.5)
        assert layer.d_p == pytest.approx(5e-6)
        assert layer.K_abs == pytest.approx(1e-12)
        assert layer.theta_contact == pytest.approx(110.0)
        assert layer.bulk_electrical_conductivity == pytest.approx(1e4)
        assert layer.bulk_specific_heat_capacity == pytest.approx(800.0)
        assert layer.bulk_thermal_conductivity == pytest.approx(0.5)
        assert layer.eps_ion == pytest.approx(0.3)

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
        # scalar field → shape (n_layers, 1)
        assert arr.shape == (len(default_cell.layers), 1)

    def test_build_property_arrays_sets_layer_fields(self, default_cell):
        """build_property_arrays sets numeric Layer fields as ndarray attributes on cell."""
        from dataclasses import fields
        for f in fields(Layer):
            val = getattr(default_cell, f.name, None)
            if isinstance(val, np.ndarray):  # only numeric fields are converted
                assert val.ndim >= 1, f"Cell.{f.name} should be at least 1-D"

    def test_build_property_arrays_scalar_shape(self, default_cell):
        """Scalar layer fields produce (n_layers, 1) arrays."""
        n = len(default_cell.layers)
        assert default_cell.thickness.shape == (n, 1)
        assert default_cell.bulk_thermal_conductivity.shape == (n, 1)

    def test_build_property_arrays_length(self, default_cell):
        """Every numeric per-layer array has n_layers in one of its dimensions."""
        from dataclasses import fields
        n_layers = len(default_cell.layers)
        for f in fields(Layer):
            val = getattr(default_cell, f.name, None)
            if isinstance(val, np.ndarray):
                assert n_layers in val.shape, \
                    f"Cell.{f.name} shape {val.shape} doesn't contain n_layers={n_layers}"

    def test_custom_name(self, custom_cell):
        assert custom_cell.name == "custom"

    def test_get_property_array_ragged_arrays(self):
        custom_ionomer = PFSAIonomer(
            rho_dry_ion=2.0e3,
            EW_ion=1100,
            darken_num_ion=np.array([0., 67.74, -32.03, 3.842]),
            darken_den_ion=np.array([103.37, -33.013, -2.115, 1.0]),
            sorption_coeffs_ion=np.array([0.043, 17.81, -39.85, 36.0]),
            lmbd_liq_ref_ion=22,
            D_lmbd_ref_ion=0.314 * 2.72e-5 * 1e-4,
            k_des_ref_ion=0.0211 * 4.59e-5,
            E_act_ion=2.54 * 20e6,
            sigma_ref_ion=50.,
            f_v_perc_ion=0.1,
            n_sigma_ion=1.5,
            T_ref_sigma_ion=298.15,
            T_ref_D_ion=303.15,
            T_ref_des_ion=303.15,
        )
        cl = mrpd.PtCCatalystLayer(
            thickness=10e-6,
            bulk_density=2010.,
            bulk_specific_heat_capacity=710.,
            bulk_thermal_conductivity=0.25,
            L_Pt=0.3e-2,
            wt_Pt=0.4,
            ic_ratio=0.7,
            ecsa=45e3,
            ionomer=mrpd.Nafion_N21X,
            r_C=25e-9,
            K_abs=1e-13,
            theta_contact=95,
            reaction=ElectrochemicalReaction(
                reference_exchange_current_density=2.47e-8,
                activation_energy=67e6,
                reaction_order=0.54,
                reference_activity=1e5,
                reference_temperature=353.15,
                number_of_electrons=2,
                charge_transfer_coeff=0.5,
            ),
        )
        ca = CellSide(name='cathode', cl=cl)
        an = CellSide(name='anode', cl=cl)
        memb = PFSAMembrane(thickness=50e-6, ionomer=custom_ionomer)
        cell = Cell(name='ragged', ca=ca, an=an, memb=memb)
        arr = cell.get_property_array('darken_num_ion')
        assert arr.ndim == 2
        assert arr.shape[1] == max(len(mrpd.Nafion_N21X.darken_num_ion),
                                   len(custom_ionomer.darken_num_ion))


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
