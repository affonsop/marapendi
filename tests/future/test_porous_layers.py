from marapendi.future.porous_layers import GasDiffusionLayer, MicroPorousLayer, PorousLayer


def test_porous_layer_defaults():
    layer = PorousLayer()
    assert layer.thickness == 1e-3
    assert layer.porosity == 1.


def test_gas_diffusion_layer_defaults():
    gdl = GasDiffusionLayer()
    assert gdl.thickness == 200e-6
    assert gdl.porosity == 0.6


def test_micro_porous_layer_defaults():
    mpl = MicroPorousLayer()
    assert mpl.thickness == 30e-6
    assert mpl.porosity == 0.4


def test_thermal_resistance():
    layer = PorousLayer(thickness=1e-3, thermal_conductivity=0.5)
    assert layer.thermal_resistance() == 1e-3 / 0.5
