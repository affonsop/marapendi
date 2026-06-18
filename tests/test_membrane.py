"""Tests for Membrane and PFSA (components/membrane/membrane/membrane.py)."""
import numpy as np
import pytest
import marapendi as mrpd
from marapendi.thermo.water import water_molar_volume


@pytest.fixture
def pfsa_thick():
    return mrpd.PFSA(
        equivalent_weight=1100,
        dry_density=1980,
        dry_thickness=125e-6,
    )


@pytest.fixture
def pfsa_thin():
    return mrpd.PFSA(
        equivalent_weight=1100,
        dry_density=1980,
        dry_thickness=25e-6,
    )


class TestMembraneWaterProperties:
    def test_water_vol_fraction_increases_with_content(self, pfsa_thick):
        fv5 = pfsa_thick.water_vol_fraction(5, water_molar_volume(353.15))
        fv15 = pfsa_thick.water_vol_fraction(15, water_molar_volume(353.15))
        assert fv15 > fv5

    def test_water_vol_fraction_known_value(self, pfsa_thin):
        # At λ=10, roughly 0.25 vol fraction for typical Nafion
        fv = pfsa_thin.water_vol_fraction(10, water_molar_volume(353.15))
        assert 0.1 < fv < 0.5

    def test_equilibrium_water_content_increases_with_rh(self, pfsa_thick):
        wc_low = pfsa_thick.equilibrium_water_content(0.3, 353.15)
        wc_high = pfsa_thick.equilibrium_water_content(0.9, 353.15)
        assert wc_high > wc_low


class TestMembraneProtonConductivity:
    def test_proton_conductivity_with_profile(self, pfsa_thick):
        # proton_conductivity expects a 1D array (water content profile across membrane)
        profile = np.linspace(8., 12., 10)
        sigma = pfsa_thick.proton_conductivity(profile, 353.15)
        assert sigma > 0

    def test_proton_conductivity_higher_water_content_means_higher_conductivity(self, pfsa_thick):
        low_profile = np.full(10, 5.)
        high_profile = np.full(10, 15.)
        sigma_low = pfsa_thick.proton_conductivity(low_profile, 353.15)
        sigma_high = pfsa_thick.proton_conductivity(high_profile, 353.15)
        assert sigma_high > sigma_low

    def test_liquid_equilibrium_water_content_increases_with_temperature(self, pfsa_thick):
        wc_low = pfsa_thick.liquid_equilibrium_water_content(300.)
        wc_high = pfsa_thick.liquid_equilibrium_water_content(353.15)
        assert wc_high > wc_low


class TestHydrogenPermeation:
    def test_hydrogen_permeation_flux_positive(self, pfsa_thick):
        flux = pfsa_thick.hydrogen_permeation_flux(
            partial_pressure_h2=1e5,
            temperature=353.15,
            pressure_difference=0.,
            water_vol_fraction=0.3,
        )
        assert flux >= 0

    def test_flux_increases_with_partial_pressure(self, pfsa_thick):
        flux_low = pfsa_thick.hydrogen_permeation_flux(5e4, 353.15, 0., 0.3)
        flux_high = pfsa_thick.hydrogen_permeation_flux(1e5, 353.15, 0., 0.3)
        assert flux_high > flux_low

    def test_thin_membrane_higher_flux_than_thick(self, pfsa_thin, pfsa_thick):
        args = dict(partial_pressure_h2=1e5, temperature=353.15,
                    pressure_difference=0., water_vol_fraction=0.3)
        flux_thin = pfsa_thin.hydrogen_permeation_flux(**args)
        flux_thick = pfsa_thick.hydrogen_permeation_flux(**args)
        assert flux_thin > flux_thick


class TestEquilibriumWaterContent:
    def test_basic_formula(self, pfsa_thin):
        rh = 0.5
        result = pfsa_thin.equilibrium_water_content(rh, 303.15)
        # Check it gives a positive number in a physically reasonable range
        assert 0 < result < 30

    def test_consistent_with_sorption_isotherm(self, pfsa_thin):
        rh_vals = np.array([0.2, 0.5, 0.8])
        wcs = [pfsa_thin.equilibrium_water_content(rh, 353.15) for rh in rh_vals]
        # Monotonically increasing
        assert wcs[0] < wcs[1] < wcs[2]
