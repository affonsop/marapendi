from marapendi.future.catalyst_layers import CatalystLayer, PtCCatalystLayer


def test_catalyst_layer_defaults():
    cl = CatalystLayer()
    assert cl.catalyst_loading == 0.2e-6 * 1e4
    assert cl.ecsa == 70e3


def test_ptc_catalyst_layer_post_init():
    cl = PtCCatalystLayer()
    carbon_loading = cl.platinum_loading * (1 / cl.catalyst_platinum_weight_percent - 1)
    assert cl.carbon_vol_fraction == carbon_loading / (cl.thickness * cl.carbon_density)
    assert cl.catalyst_vol_fraction > 0
    assert cl.dry_ionomer_vol_fraction > 0
    assert cl.carbon_agglomerate_number_density > 0
