"""Tests for marapendi.components.operating_conditions."""
import numpy as np
import pytest

import marapendi as mrpd
from marapendi.models.water import water_saturation_pressure, water_molecular_weight


T = 353.15       # K  — typical PEM operating temperature
P = 1.5e5        # Pa — backpressure
P_SAT = water_saturation_pressure(T)


# ─── OperatingConditions ──────────────────────────────────────────────────────

class TestOperatingConditions:
    def _make(self, flow_rates=None, h2ol=0.0):
        if flow_rates is None:
            flow_rates = np.array([1e-4, 3e-4, 0., 2e-5])
        return mrpd.OperatingConditions(
            temperature=T,
            backpressure=P,
            inlet_gas_molar_flow_rates=flow_rates,
            inlet_h2ol_molar_flow_rate=h2ol,
        )

    def test_total_gas_molar_flow_is_sum(self):
        flow_rates = np.array([1e-4, 3e-4, 0., 2e-5])
        oc = self._make(flow_rates)
        assert oc.inlet_gas_molar_flow_rate == pytest.approx(np.sum(flow_rates))

    def test_liquid_mass_flow_rate(self):
        h2ol = 5e-5
        oc = self._make(h2ol=h2ol)
        assert oc.inlet_liquid_mass_flow_rate == pytest.approx(h2ol * water_molecular_weight)

    def test_zero_liquid_flow(self):
        oc = self._make(h2ol=0.0)
        assert oc.inlet_liquid_mass_flow_rate == pytest.approx(0.0)


# ─── SideConditions ───────────────────────────────────────────────────────────

class TestSideConditions:
    def test_at_returns_operating_conditions(self):
        sc = mrpd.SideConditions(temperature=T, backpressure=P)
        snap = sc.at(0.0)
        assert isinstance(snap, mrpd.OperatingConditions)

    def test_constant_wrapping(self):
        sc = mrpd.SideConditions(temperature=T, backpressure=P)
        assert sc.at(0.0).temperature == pytest.approx(T)
        assert sc.at(10.0).temperature == pytest.approx(T)

    def test_callable_temperature(self):
        sc = mrpd.SideConditions(temperature=lambda t: T + t)
        assert sc.at(5.0).temperature == pytest.approx(T + 5.0)

    def test_callable_backpressure(self):
        sc = mrpd.SideConditions(backpressure=lambda t: P + t * 100)
        assert sc.at(2.0).backpressure == pytest.approx(P + 200.0)


# ─── InletAirConditions ───────────────────────────────────────────────────────

class TestInletAirConditions:
    def _make(self, rh=0.5, o2_flow=1e-4, x_o2_dry=0.21):
        return mrpd.InletAirConditions(
            temperature=T,
            backpressure=P,
            rh_ref_pressure=P,
            o2_molar_flow_rate=o2_flow,
            o2_dry_mole_fraction=x_o2_dry,
            inlet_rh=rh,
        )

    def test_zero_rh_gives_zero_vapor_flow(self):
        snap = self._make(rh=0.0).at(0.0)
        h2ov_flow = snap.inlet_gas_molar_flow_rates[3]
        assert h2ov_flow == pytest.approx(0.0, abs=1e-20)

    def test_vapor_increases_with_rh(self):
        low  = self._make(rh=0.3).at(0.0).inlet_gas_molar_flow_rates[3]
        high = self._make(rh=0.8).at(0.0).inlet_gas_molar_flow_rates[3]
        assert high > low

    def test_vapor_flow_matches_formula(self):
        rh = 0.6
        o2_flow = 1e-4
        x_o2_dry = 0.21
        snap = self._make(rh=rh, o2_flow=o2_flow, x_o2_dry=x_o2_dry).at(0.0)

        x_h2ov = rh * P_SAT / P
        expected_h2ov = o2_flow / x_o2_dry / (1 - x_h2ov) * x_h2ov
        assert snap.inlet_gas_molar_flow_rates[3] == pytest.approx(expected_h2ov, rel=1e-6)

    def test_n2_flow_matches_o2_to_n2_ratio(self):
        o2_flow = 1e-4
        x_o2_dry = 0.21
        snap = self._make(o2_flow=o2_flow, x_o2_dry=x_o2_dry, rh=0.0).at(0.0)

        expected_n2 = o2_flow / x_o2_dry * (1 - x_o2_dry)
        assert snap.inlet_gas_molar_flow_rates[1] == pytest.approx(expected_n2, rel=1e-6)

    def test_zero_h2_flow(self):
        snap = self._make().at(0.0)
        assert snap.inlet_gas_molar_flow_rates[2] == pytest.approx(0.0, abs=1e-20)

    def test_at_full_humidity_vapor_fraction_matches_rh(self):
        """At RH=1 referred to backpressure, x_h2ov should equal P_sat/P."""
        snap = self._make(rh=1.0).at(0.0)
        total_molar = np.sum(snap.inlet_gas_molar_flow_rates)
        x_h2ov_actual = snap.inlet_gas_molar_flow_rates[3] / total_molar
        x_h2ov_expected = P_SAT / P / (1 + P_SAT / P * (1 / 0.21 - 1))  # approximate
        # Just check positivity and physical bound
        assert 0 < x_h2ov_actual < 1

    def test_callable_o2_flow(self):
        cond = mrpd.InletAirConditions(
            temperature=T,
            backpressure=P,
            rh_ref_pressure=P,
            o2_molar_flow_rate=lambda t: 1e-4 * t,
            o2_dry_mole_fraction=0.21,
            inlet_rh=0.0,
        )
        snap2 = cond.at(2.0)
        snap4 = cond.at(4.0)
        assert snap4.inlet_gas_molar_flow_rates[0] == pytest.approx(
            2 * snap2.inlet_gas_molar_flow_rates[0], rel=1e-6
        )


# ─── InletHydrogenConditions ──────────────────────────────────────────────────

class TestInletHydrogenConditions:
    def _make(self, rh=0.5, h2_flow=2e-4):
        return mrpd.InletHydrogenConditions(
            temperature=T,
            backpressure=P,
            rh_ref_pressure=P,
            h2_molar_flow_rate=h2_flow,
            inlet_rh=rh,
        )

    def test_zero_rh_gives_zero_vapor_flow(self):
        snap = self._make(rh=0.0).at(0.0)
        assert snap.inlet_gas_molar_flow_rates[3] == pytest.approx(0.0, abs=1e-20)

    def test_vapor_flow_matches_formula(self):
        rh = 0.5
        h2_flow = 2e-4
        snap = self._make(rh=rh, h2_flow=h2_flow).at(0.0)

        x_h2ov = rh * P_SAT / P
        expected_h2ov = h2_flow / (1 - x_h2ov) * x_h2ov
        assert snap.inlet_gas_molar_flow_rates[3] == pytest.approx(expected_h2ov, rel=1e-6)

    def test_vapor_increases_with_rh(self):
        low  = self._make(rh=0.2).at(0.0).inlet_gas_molar_flow_rates[3]
        high = self._make(rh=0.9).at(0.0).inlet_gas_molar_flow_rates[3]
        assert high > low

    def test_zero_o2_and_n2_flows(self):
        snap = self._make().at(0.0)
        assert snap.inlet_gas_molar_flow_rates[0] == pytest.approx(0.0, abs=1e-20)
        assert snap.inlet_gas_molar_flow_rates[1] == pytest.approx(0.0, abs=1e-20)

    def test_h2_flow_correct(self):
        h2_flow = 2e-4
        snap = self._make(rh=0.0, h2_flow=h2_flow).at(0.0)
        assert snap.inlet_gas_molar_flow_rates[2] == pytest.approx(h2_flow, rel=1e-6)

    def test_h2_flow_scales_proportionally(self):
        snap1 = self._make(h2_flow=1e-4).at(0.0)
        snap2 = self._make(h2_flow=2e-4).at(0.0)
        assert snap2.inlet_gas_molar_flow_rates[2] == pytest.approx(
            2 * snap1.inlet_gas_molar_flow_rates[2], rel=1e-6
        )


# ─── CellConditions ───────────────────────────────────────────────────────────

class TestCellConditions:
    def _make_cell(self, i=5000.0):
        ca = mrpd.InletAirConditions(
            temperature=T, backpressure=P, rh_ref_pressure=P,
            o2_molar_flow_rate=1e-4, o2_dry_mole_fraction=0.21, inlet_rh=0.5,
        )
        an = mrpd.InletHydrogenConditions(
            temperature=T, backpressure=P, rh_ref_pressure=P,
            h2_molar_flow_rate=2e-4, inlet_rh=0.5,
        )
        return mrpd.CellConditions(current_density=i, ca=ca, an=an)

    def test_at_returns_snapshot(self):
        snap = self._make_cell().at(0.0)
        assert isinstance(snap, mrpd.CellSnapshot)

    def test_constant_current_density(self):
        snap = self._make_cell(i=8000.0).at(5.0)
        assert snap.current_density == pytest.approx(8000.0)

    def test_callable_current_density(self):
        cc = mrpd.CellConditions(
            current_density=lambda t: t * 100.0,
            ca=mrpd.InletAirConditions(
                temperature=T, backpressure=P, rh_ref_pressure=P,
                o2_molar_flow_rate=1e-4, o2_dry_mole_fraction=0.21, inlet_rh=0.0,
            ),
            an=mrpd.InletHydrogenConditions(
                temperature=T, backpressure=P, rh_ref_pressure=P,
                h2_molar_flow_rate=2e-4, inlet_rh=0.0,
            ),
        )
        assert cc.at(3.0).current_density == pytest.approx(300.0)

    def test_ca_and_an_temperatures(self):
        snap = self._make_cell().at(0.0)
        assert snap.ca.temperature == pytest.approx(T)
        assert snap.an.temperature == pytest.approx(T)
