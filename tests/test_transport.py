"""Tests for marapendi.models.transport — gas and liquid transport models."""
import numpy as np
import pytest
import marapendi as mrpd

T = 353.15
P = 1.5e5


@pytest.fixture
def gas_model():
    return mrpd.PorousGasResistanceModel()


@pytest.fixture
def darcy_model():
    return mrpd.DarcyTransportModel()


# ─── PorousGasResistanceModel ─────────────────────────────────────────────────

class TestSpeciesDiffusionCoefficient:
    # species_diffusion_coefficient expects T and pressure as 2-D arrays
    # (n_layers, m) — matching how TransientCellModel passes them.

    def test_shape(self, gas_model):
        n = 1
        T_arr = np.full((n, 1), T)
        P_arr = np.full((n, 1), P)
        x_h2  = np.zeros(n)
        D = gas_model.species_diffusion_coefficient(T_arr, P_arr, x_h2)
        assert D.shape == (n, 4, 1)

    def test_positive(self, gas_model):
        n = 3
        T_arr = np.full((n, 1), T)
        P_arr = np.full((n, 1), P)
        x_h2  = np.zeros(n)
        D = gas_model.species_diffusion_coefficient(T_arr, P_arr, x_h2)
        assert np.all(D > 0)

    def test_increases_with_temperature(self, gas_model):
        n = 1
        x_h2  = np.zeros(n)
        D_cold = gas_model.species_diffusion_coefficient(np.full((n, 1), 300.), np.full((n, 1), P), x_h2)
        D_hot  = gas_model.species_diffusion_coefficient(np.full((n, 1), 400.), np.full((n, 1), P), x_h2)
        assert np.all(D_hot > D_cold)


class TestWaterSaturationCorrection:
    def test_one_at_zero_saturation(self, gas_model):
        # At s=0 the correction is 1 (no blockage)
        assert gas_model.water_saturation_correction(0., n_s=3) == pytest.approx(1.0, rel=1e-6)

    def test_decreases_with_saturation(self, gas_model):
        f0 = gas_model.water_saturation_correction(0.1, n_s=3)
        f1 = gas_model.water_saturation_correction(0.5, n_s=3)
        assert f0 > f1

    def test_clipped_not_zero(self, gas_model):
        # Never drops below (1e-6)^n_s — no division by zero downstream
        assert gas_model.water_saturation_correction(1.0, n_s=3) > 0


class TestTotalDiffusionResistance:
    def test_positive_finite(self, gas_model):
        n = 2
        T_arr = np.full((n, 1, 1), T)
        s_arr = np.full((n, 1, 1), 0.05)
        D_gk  = np.full((n, 4, 1), 1e-5)
        M_k   = np.full((n, 4, 1), 20.)
        thickness = np.full((n, 1, 1), 160e-6)
        eps_p = np.full((n, 1, 1), 0.72)
        tort  = np.full((n, 1, 1), 3.)
        d_p   = np.full((n, 1, 1), 20e-6)
        n_s   = np.full((n, 1, 1), 3.)
        R = gas_model.total_diffusion_resistance(T_arr, s_arr, D_gk, M_k,
                                                  thickness, eps_p, tort, d_p, n_s)
        assert np.all(np.isfinite(R))
        assert np.all(R > 0)


# ─── DarcyTransportModel ─────────────────────────────────────────────────────

class TestLiquidDarcyFlowResistance:
    def test_positive(self, darcy_model):
        nu_l = mrpd.water_kinematic_viscosity(T)
        R = darcy_model.calculate_liquid_darcy_flow_resistance(
            s=0.1, nu_l=nu_l, thickness=160e-6, K_abs=1e-12, n_rel=3,
        )
        assert np.all(R > 0)

    def test_increases_as_saturation_drops(self, darcy_model):
        nu_l = mrpd.water_kinematic_viscosity(T)
        R_high = darcy_model.calculate_liquid_darcy_flow_resistance(0.05, nu_l, 160e-6, 1e-12, 3)
        R_low  = darcy_model.calculate_liquid_darcy_flow_resistance(0.50, nu_l, 160e-6, 1e-12, 3)
        assert R_high > R_low


class TestCapillaryPressure:
    @pytest.mark.parametrize("s", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_positive_finite(self, darcy_model, s):
        p_c = darcy_model.capillary_pressure_from_saturation(
            s, p_b=15020, m=0.7262, n=3.652,
        )
        assert np.isfinite(p_c)
        assert p_c > 0

    def test_increases_with_saturation(self, darcy_model):
        kwargs = dict(p_b=15020, m=0.7262, n=3.652)
        p_low  = darcy_model.capillary_pressure_from_saturation(0.1, **kwargs)
        p_high = darcy_model.capillary_pressure_from_saturation(0.9, **kwargs)
        assert p_high > p_low

    def test_clipped_saturation_stays_finite(self, darcy_model):
        # s=1 should not produce inf after our guard (clip to 1-1e-6)
        p_c = darcy_model.capillary_pressure_from_saturation(1.0, 15020, 0.7262, 3.652)
        assert np.isfinite(p_c)
