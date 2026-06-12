from marapendi.membrane import Membrane, PFSA


def test_membrane_defaults():
    membrane = Membrane()
    assert membrane.dry_thickness == 25e-6
    assert membrane.dry_concentration == membrane.dry_density / membrane.equivalent_weight
    assert membrane.dry_molar_volume == 1. / membrane.dry_concentration


def test_pfsa_defaults():
    pfsa = PFSA()
    assert pfsa.conductivity_exp == 1.5
    assert pfsa.dry_thickness == 25e-6


def test_pfsa_equilibrium_water_content_increases_with_rh():
    pfsa = PFSA()
    assert pfsa.equilibrium_water_content(1., 353.15) > pfsa.equilibrium_water_content(0.5, 353.15)


def test_pfsa_equilibrium_water_content_relaxation():
    pfsa = PFSA()
    assert pfsa.equilibrium_water_content(0.5, 353.15, s_relax=1.) == (
        pfsa.equilibrium_water_content(0.5, 353.15, s_relax=0.) + 1.
    )


def test_pfsa_proton_resistance_positive():
    import numpy as np

    pfsa = PFSA()
    water_content_profile = np.full((10,), 10.)
    assert pfsa.proton_resistance(water_content_profile, 353.15) > 0


def test_membrane_water_diffusivity_and_absorption_coefficient_positive():
    membrane = Membrane()
    assert membrane.calculate_water_diffusivity(353.15) > 0
    assert membrane.calculate_water_absorption_coefficient(353.15) > 0


def test_membrane_electroosmotic_drag_speed_scales_with_current():
    membrane = Membrane()
    speed = membrane.calculate_electroosmotic_drag_speed(353.15, 1.)
    assert speed == 2 * membrane.calculate_electroosmotic_drag_speed(353.15, 0.5)
