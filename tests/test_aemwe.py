"""Tests for the AEMWE model (components/cell/aem_electrolyzer.py)."""
import pytest
import numpy as np
import marapendi as mrpd
from marapendi.components.electrolyte.koh import KOH_solution


# Pre-defined KOH instances for electrolyte tests
KOH_1M = KOH_solution(molality=1.)
KOH_20_wt_percent = KOH_solution(weight_percent=20)
KOH_45_wt_percent = KOH_solution(weight_percent=45)


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
            absolute_permeability=1e-12,
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
    return mrpd.PAP85(
        dry_thickness=80e-6,
        water_balance_model=mrpd.MembraneWaterBalanceModel(),
    )


@pytest.fixture
def electrolyzer_cell(cathode, anode, membrane):
    return mrpd.ElectrolyzerCell(
        ca=cathode, an=anode, membrane=membrane,
        area=5e-4, cell_number=1, electrical_resistance=60e-7,
    )


# ---------------------------------------------------------------------------
# Electrochemistry
# ---------------------------------------------------------------------------

def test_reversible_cell_voltage():
    assert np.isclose(mrpd.calculate_reversible_cell_voltage(298.15, 1), 1.229, rtol=1e-3)


# ---------------------------------------------------------------------------
# Electrolyte
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Cell structure
# ---------------------------------------------------------------------------

def test_electrolyzer_cell_construction(electrolyzer_cell):
    assert electrolyzer_cell.ca is not None
    assert electrolyzer_cell.an is not None
    assert electrolyzer_cell.membrane is not None


def test_cathode_pap_ionomer(cathode):
    assert isinstance(cathode.cl.ionomer, mrpd.PAPIonomer)


def test_membrane_pap85(membrane):
    assert isinstance(membrane, mrpd.PAP85)
    assert membrane.dry_thickness == 80e-6


def test_porous_transfer_layer_ptl_porosity(anode):
    assert np.isclose(anode.cl.ptl_porosity, 0.83)
