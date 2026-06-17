"""Tests for ionomer classes (components/membrane/ionomer/)."""
import numpy as np
import pytest
import marapendi as mrpd


@pytest.fixture
def pfsa():
    return mrpd.PFSAIonomer(
        dry_density=2004.,
        equivalent_weight=952.,
        conductivity_correction=1.0,
        conductivity_exp=1.5,
    )


@pytest.fixture
def pap():
    return mrpd.PAPIonomer()


class TestIonomerBase:
    def test_dry_concentration_from_density_and_ew(self, pfsa):
        expected = pfsa.dry_density / pfsa.equivalent_weight
        assert np.isclose(pfsa.dry_concentration, expected)

    def test_water_vol_fraction_increases_with_content(self, pfsa):
        from marapendi.models.water import water_molar_volume
        fv_low = pfsa.water_vol_fraction(5, water_molar_volume(353.15))
        fv_high = pfsa.water_vol_fraction(20, water_molar_volume(353.15))
        assert fv_high > fv_low

    def test_wet_expansion_factor_gte_one(self, pfsa):
        factor = pfsa.wet_expansion_factor(10, 353.15)
        assert factor >= 1.0


class TestPFSAIonomer:
    def test_equilibrium_water_content_increases_with_rh(self, pfsa):
        wc_low = pfsa.equilibrium_water_content(0.3)
        wc_high = pfsa.equilibrium_water_content(0.9)
        assert wc_high > wc_low

    def test_proton_conductivity_positive(self, pfsa):
        sigma = pfsa.proton_conductivity(10, temperature=353.15)
        assert sigma > 0

    def test_proton_conductivity_increases_with_water_content(self, pfsa):
        sigma_low = pfsa.proton_conductivity(5, temperature=353.15)
        sigma_high = pfsa.proton_conductivity(15, temperature=353.15)
        assert sigma_high > sigma_low

    def test_o2_permeability_increases_with_water_content(self, pfsa):
        perm_low = pfsa.o2_permeability(5, temperature=353.15)
        perm_high = pfsa.o2_permeability(15, temperature=353.15)
        assert perm_high > perm_low

    def test_o2_diffusion_coefficient_order_of_magnitude(self, pfsa):
        D = pfsa.o2_film_diffusion_coefficient(10, temperature=353.15)
        # Should be in the ~1e-10 m2/s range for ionomer films
        assert 1e-12 < D < 1e-8

    def test_nafion_d2020_instance(self):
        assert isinstance(mrpd.NafionD2020, mrpd.PFSAIonomer)
        assert np.isclose(mrpd.NafionD2020.equivalent_weight, 952.)


class TestPAPIonomer:
    def test_pap_instantiates(self, pap):
        assert isinstance(pap, mrpd.PAPIonomer)

    def test_pap_conductivity_positive(self, pap):
        # PAP ionomer should provide hydroxide conductivity
        sigma = pap.hydroxide_conductivity(10, temperature=353.15)
        assert sigma > 0
