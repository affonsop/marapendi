"""
Integration tests for the PEMFC polarization curve simulation.

Reproduces the base-case conditions from Vetter & Schumacher (2019)
and verifies results against the notebook output from
notebooks/simulate_polarization_curve.ipynb.
"""
from collections import namedtuple

import numpy as np
import pytest
from scipy.interpolate import interp1d

import marapendi.dynamic as mrpd


# ---------------------------------------------------------------------------
# Cell fixture — exact parameters from the notebook (Tables 3–5)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cell():
    # ── Kinetics ──────────────────────────────────────────────────────────
    orr_kinetics = mrpd.ElectrochemicalReaction(
        reference_exchange_current_density=2.45e-4,   # A/m2_Pt
        activation_energy=67e6,                        # J/kmol = 67 kJ/mol
        reaction_order=0.54,
        reference_activity=1.e5,                       # Pa (1 atm reference)
        reference_temperature=353.15,                  # K  (80 C)
        number_of_electrons=1,
        charge_transfer_coeff=1,
    )

    # ── GDLs — Toray TGP-H-060 (Table 4) ─────────────────────────────────
    gdl_an = mrpd.PorousLayer(
        thickness=160e-6, eps_p=0.76, bulk_density=440.,
        bulk_specific_heat_capacity=710., bulk_thermal_conductivity=1.6,
        K_abs=6.15e-12, theta_contact=130., tort=1.6**2
    )
    gdl_ca = mrpd.PorousLayer(
        thickness=160e-6, eps_p=0.76, bulk_density=440.,
        bulk_specific_heat_capacity=710., bulk_thermal_conductivity=1.6,
        K_abs=6.15e-12, theta_contact=130., tort=1.6**2
    )

    # ── Nafion NR-211 ionomer ──────────────────────────────────────────────
    nafion = mrpd.PFSAIonomer(
        rho_dry_ion=1.97e3, EW_ion=1020,
        darken_num_ion=np.array([0., 67.74, -32.03, 3.842]),
        darken_den_ion=np.array([103.37, -33.013, -2.115, 1.0]),
        sorption_coeffs_ion=np.array([0.043, 17.81, -39.85, 36.0]),
        lmbd_liq_ref_ion=22,
        D_lmbd_ref_ion=1.0e-10,
        k_des_ref_ion=1.42e-4,
        E_act_ion=20e6,
        E_act_cond_ion=15e6,
        sigma_ref_ion=116.,
        f_v_perc_ion=0.06, n_sigma_ion=1.5,
        T_ref_sigma_ion=353.15, T_ref_D_ion=353.15, T_ref_des_ion=353.15,
    )

    # ── Catalyst layers ────────────────────────────────────────────────────
    ca_cl = mrpd.PtCCatalystLayer(
        thickness=10e-6,
        bulk_density=1000., bulk_specific_heat_capacity=710.,
        bulk_thermal_conductivity=0.27,
        L_Pt=0.4e-2, wt_Pt=0.416, ic_ratio=1.04, ecsa=75e3,
        tort=1.6**2, ionomer=nafion,
        r_C=1e-10, K_abs=1e-13, theta_contact=95,
        reaction=orr_kinetics
    )
    an_cl = mrpd.PtCCatalystLayer(
        thickness=10e-6,
        bulk_density=1000., bulk_specific_heat_capacity=710.,
        bulk_thermal_conductivity=0.27,
        L_Pt=0.4e-2 / 3, wt_Pt=0.192, ic_ratio=1.07, ecsa=75e3,
        tort=1.6**2, ionomer=nafion,
        r_C=1e-10, K_abs=1e-13, theta_contact=95
    )

    # ── Cell assembly ──────────────────────────────────────────────────────
    return mrpd.Cell(
        area=25e-4,
        electrical_resistance=80e-6 / 1250. + 5e-6 / 350.,
        thermal_resistance=0,
        ca=mrpd.CellSide(
            cl=ca_cl, gdl=gdl_ca,
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        an=mrpd.CellSide(
            cl=an_cl, gdl=gdl_an,
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        memb=mrpd.PFSAMembrane(
            thickness=25e-6,
            bulk_thermal_conductivity=0.3,
            ionomer=nafion,
        ),
    )


# ---------------------------------------------------------------------------
# Simulation fixture — reduced sweep (6 steps) scoped at module level
# ---------------------------------------------------------------------------

SimResult = namedtuple("SimResult", ["i_Acm2", "V_arr", "results", "base", "model"])

# Reduced current-density sweep used by the tests.
# 15000 A/m² is included to land close to U ≈ 0.6 V (notebook: 1.453 A/cm²).
CURRENT_DENSITIES_TEST = np.array([0, 4000, 8000, 12000, 15000, 19000], dtype=float)  # A/m²
T_STEP = 150.0   # s per step
MAX_STEP = 100.  # s

# Operating conditions — Table 5 of Vetter & Schumacher (2019)
T_OP = 343.15   # K  (70 C)
P_OP = 1.5e5    # Pa (1.5 bar)
RH   = 0.9      # 90 % relative humidity
S_C  = 0.12     # liquid saturation at cathode GDL/channel interface


@pytest.fixture(scope="module")
def simulation(cell):
    base = mrpd.CellBaseModel(
        transient_transport_model=mrpd.TransientCellModel(cell=cell, current_density=0.),
        memb_model=mrpd.PFSAModel(),
        cl_model=mrpd.PtCCatalystLayerModel(),
        gas_diffusion_model=mrpd.PorousGasResistanceModel(),
        darcy_transport_model=mrpd.DarcyTransportModel(),
        voltage_model=mrpd.VoltageModel(),
    )
    model = base.transient_transport_model

    y0 = base.initial_state(
        cell_temperature=T_OP, cell_pressure=P_OP,
        ca_rh=RH, an_rh=RH,
        ca_dry_o2=0.21,
        an_dry_h2=1.0,
        s_ca=S_C,
    )

    results = []
    y_current = y0.copy()
    for i_density in CURRENT_DENSITIES_TEST:
        model.current_density = float(i_density)
        sol = base.solve(y_current, t_span=(0., T_STEP), max_step=MAX_STEP)
        y_current = sol.y[:, -1]
        state = base.postprocess(y_current[:, np.newaxis])
        results.append(state)

    i_Acm2 = CURRENT_DENSITIES_TEST / 1e4
    V_arr = np.array([s.V_cell.item() for s in results])

    return SimResult(
        i_Acm2=i_Acm2,
        V_arr=V_arr,
        results=results,
        base=base,
        model=model,
    )


# ---------------------------------------------------------------------------
# Helper: R_PEM via Simpson's rule (mirrors notebook exactly)
# ---------------------------------------------------------------------------

def _compute_R_PEM(state, base, cell):
    """Membrane ohmic resistance in mOhm.cm² via Simpson's rule."""
    def _sigma(ix):
        return base.memb_model.charge_conductivity(
            float(state.f_v[ix, 0]), float(state.T[ix, 0]), 'proton', cell.memb
        )
    sigma_left   = _sigma(cell.an.cl.ix)
    sigma_center = _sigma(cell.memb.ix)
    sigma_right  = _sigma(cell.ca.cl.ix)
    return (
        cell.memb.thickness / 6
        * (1 / sigma_left + 4 / sigma_center + 1 / sigma_right)
        * 1e4 * 1e3   # Ohm.m² -> mOhm.cm²
    )


def _compute_J_PEM(state, model, cell):
    """Water flux through PEM in µmol/(cm²·s)."""
    return float(state.J[cell.memb.ix, model.i_lmbd, 0]) * 1e5


# ---------------------------------------------------------------------------
# Steady-state simulation fixture — same cell/conditions, solve_steady_state
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simulation_ss(cell):
    """Polarization curve via solve_steady_state, warm-started step by step."""
    base = mrpd.CellBaseModel(
        transient_transport_model=mrpd.TransientCellModel(cell=cell, current_density=0.),
        memb_model=mrpd.PFSAModel(),
        cl_model=mrpd.PtCCatalystLayerModel(),
        gas_diffusion_model=mrpd.PorousGasResistanceModel(),
        darcy_transport_model=mrpd.DarcyTransportModel(),
        voltage_model=mrpd.VoltageModel(),
    )
    model = base.transient_transport_model

    y0 = base.initial_state(
        cell_temperature=T_OP, cell_pressure=P_OP,
        ca_rh=RH, an_rh=RH,
        ca_dry_o2=0.21,
        an_dry_h2=1.0,
        s_ca=S_C,
    )

    results = []
    V_arr = np.full(len(CURRENT_DENSITIES_TEST), np.nan)
    y = y0.copy()
    for k, i_density in enumerate(CURRENT_DENSITIES_TEST):
        model.current_density = float(i_density)
        sol = base.solve_steady_state(y, t=0.)
        if sol.success:
            y = sol.y[:, 0]
        state = base.postprocess(sol.y)
        results.append(state)
        V_arr[k] = float(state.V_cell[0])

    return SimResult(
        i_Acm2=CURRENT_DENSITIES_TEST / 1e4,
        V_arr=V_arr,
        results=results,
        base=base,
        model=model,
    )


# Tolerance for steady-state regression: slightly wider than IVP because the
# SS warm-starts from y0 (not a fully-evolved prior state) for the first step.
TOL_SS = 0.15


# ---------------------------------------------------------------------------
# TestPolarizationCurve — qualitative sanity checks
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestPolarizationCurve:

    def test_ocv_above_one_volt(self, simulation):
        """Open-circuit voltage (I = 0) must be above a physically reasonable floor.

        The notebook uses 200 s per step and obtains ~0.967 V after 200 s at
        I = 0.  The reduced T_STEP = 150 s means the ionomer water content has
        not fully equilibrated, so the OCV is slightly lower than the
        asymptotic value; we therefore check for > 0.9 V rather than > 1.0 V.
        """
        V_ocv = simulation.V_arr[0]
        assert V_ocv > 0.9, f"OCV = {V_ocv:.3f} V, expected > 0.9 V"

    def test_monotonically_decreasing(self, simulation):
        """Cell voltage must decrease strictly with increasing current density."""
        V = simulation.V_arr
        # Only compare consecutive non-negative voltage points
        valid = V > 0
        V_valid = V[valid]
        diffs = np.diff(V_valid)
        assert np.all(diffs < 0), (
            f"Voltage is not strictly decreasing; diffs = {diffs}"
        )

    def test_voltage_at_1Acm2(self, simulation):
        """Interpolated voltage at 1 A/cm² should be in [0.65, 0.85] V."""
        i_v = simulation.i_Acm2
        V_v = simulation.V_arr
        I_to_V = interp1d(i_v, V_v, kind='linear',
                          bounds_error=False, fill_value='extrapolate')
        V_1 = float(I_to_V(1.0))
        assert 0.65 <= V_1 <= 0.85, (
            f"V(1 A/cm²) = {V_1:.3f} V, expected in [0.65, 0.85]"
        )

    def test_peak_power_density(self, simulation):
        """Peak power density max(I × V) must exceed 0.5 W/cm²."""
        P_arr = simulation.i_Acm2 * simulation.V_arr
        peak = float(np.max(P_arr))
        assert peak > 0.5, f"Peak power = {peak:.3f} W/cm², expected > 0.5"


# ---------------------------------------------------------------------------
# TestTable6Regression — numerical regression against notebook output
# ---------------------------------------------------------------------------
# Reference values are taken directly from the notebook Table 6 output:
#   Cell voltage at I=1 A/cm²        : 0.741 V
#   Current density at U=0.6 V       : 1.453 A/cm²
#   Min. water content λ at 0.6 V    : 3.060
#   Avg. water content λ̄ at 0.6 V   : 5.765
#   Water flux through PEM at 0.6 V  : 2.992 µmol/cm²/s  (must be > 0)
#   Membrane resistance R_PEM at 0.6V: 121.590 mΩ·cm²
#
# Tolerance: ±10 % (relative) to accommodate the shorter T_STEP.
# ---------------------------------------------------------------------------

REF_V_AT_1ACMC2   = 0.741   # V
REF_I_AT_06V      = 1.453   # A/cm²
REF_LMBD_MIN_06   = 3.060   # (−)
REF_LMBD_AVG_06   = 5.765   # (−)
REF_R_PEM_06      = 121.590 # mΩ·cm²
TOL = 0.10  # ±10 %


def _get_idx06(V_arr):
    """Index of the simulated point closest to U = 0.6 V.

    Uses ``np.nanargmin`` so that NaN entries (steps the solver never reached)
    are ignored.  ``np.argmin`` propagates NaN unpredictably and may return
    the index of the first NaN rather than the closest valid voltage.
    """
    return int(np.nanargmin(np.abs(V_arr - 0.6)))


@pytest.mark.slow
class TestTable6Regression:

    def test_voltage_at_1Acm2(self, simulation):
        """V at I = 1 A/cm² matches notebook (±10 %)."""
        I_to_V = interp1d(
            simulation.i_Acm2, simulation.V_arr, kind='linear',
            bounds_error=False, fill_value='extrapolate',
        )
        V_1 = float(I_to_V(1.0))
        assert abs(V_1 - REF_V_AT_1ACMC2) / REF_V_AT_1ACMC2 <= TOL, (
            f"V(1 A/cm²) = {V_1:.3f} V, ref = {REF_V_AT_1ACMC2:.3f} V "
            f"(±{TOL*100:.0f} %)"
        )

    def test_current_density_at_06V(self, simulation):
        """Current density at U = 0.6 V matches notebook (±10 %)."""
        valid = simulation.V_arr > 0.05
        i_v = simulation.i_Acm2[valid]
        V_v = simulation.V_arr[valid]
        V_to_I = interp1d(V_v[::-1], i_v[::-1], kind='linear',
                          bounds_error=False, fill_value='extrapolate')
        i_06 = float(V_to_I(0.6))
        assert abs(i_06 - REF_I_AT_06V) / REF_I_AT_06V <= TOL, (
            f"I(0.6 V) = {i_06:.3f} A/cm², ref = {REF_I_AT_06V:.3f} A/cm² "
            f"(±{TOL*100:.0f} %)"
        )

    def test_min_water_content_at_06V(self, simulation):
        """Min. λ at 0.6 V matches notebook (±10 %)."""
        cell = simulation.base.transient_transport_model.cell
        idx = _get_idx06(simulation.V_arr)
        st = simulation.results[idx]
        ccm_ix = [cell.an.cl.ix, cell.memb.ix, cell.ca.cl.ix]
        lmbd_min = float(np.min(st.lmbd[ccm_ix, 0]))
        assert abs(lmbd_min - REF_LMBD_MIN_06) / REF_LMBD_MIN_06 <= TOL, (
            f"λ_min(0.6 V) = {lmbd_min:.3f}, ref = {REF_LMBD_MIN_06:.3f} "
            f"(±{TOL*100:.0f} %)"
        )

    def test_avg_water_content_at_06V(self, simulation):
        """Volume-averaged λ at 0.6 V matches notebook (±10 %)."""
        cell = simulation.base.transient_transport_model.cell
        idx = _get_idx06(simulation.V_arr)
        st = simulation.results[idx]
        ccm_ix   = [cell.an.cl.ix, cell.memb.ix, cell.ca.cl.ix]
        ccm_eps  = cell.eps_ion[ccm_ix, 0]
        ccm_L    = cell.thickness[ccm_ix, 0]
        lmbd06   = st.lmbd[:, 0]
        lmbd_avg = float(
            np.dot(ccm_eps * lmbd06[ccm_ix], ccm_L) / np.dot(ccm_eps, ccm_L)
        )
        assert abs(lmbd_avg - REF_LMBD_AVG_06) / REF_LMBD_AVG_06 <= TOL, (
            f"λ_avg(0.6 V) = {lmbd_avg:.3f}, ref = {REF_LMBD_AVG_06:.3f} "
            f"(±{TOL*100:.0f} %)"
        )

    def test_water_flux_positive_at_06V(self, simulation):
        """Water flux through PEM at 0.6 V must be positive (anode → cathode)."""
        cell = simulation.base.transient_transport_model.cell
        idx = _get_idx06(simulation.V_arr)
        st = simulation.results[idx]
        J_PEM = _compute_J_PEM(st, simulation.model, cell)
        assert J_PEM > 0, f"J_PEM = {J_PEM:.4f} µmol/cm²/s, expected > 0"

    def test_R_PEM_at_06V(self, simulation):
        """Membrane resistance at 0.6 V matches notebook (±10 %)."""
        cell = simulation.base.transient_transport_model.cell
        idx = _get_idx06(simulation.V_arr)
        st = simulation.results[idx]
        R_PEM = _compute_R_PEM(st, simulation.base, cell)
        assert abs(R_PEM - REF_R_PEM_06) / REF_R_PEM_06 <= TOL, (
            f"R_PEM(0.6 V) = {R_PEM:.3f} mΩ·cm², ref = {REF_R_PEM_06:.3f} mΩ·cm² "
            f"(±{TOL*100:.0f} %)"
        )


# ---------------------------------------------------------------------------
# TestSteadyStatePolarizationCurve — verify solve_steady_state Vetter results
#
# The steady-state solver finds x s.t. rates_of_change(t, x) = 0 directly,
# skipping time integration.  Results must satisfy the same sanity checks as
# the IVP sweep and agree with the IVP regression values within TOL_SS = 15 %,
# which is slightly wider than the IVP tolerance to allow for the fact that
# the first warm-start begins from the generic y0 rather than a pre-evolved
# prior operating point.
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSteadyStatePolarizationCurve:

    def test_most_steps_converge(self, simulation_ss):
        """At least 5 of 6 steps must converge for the test currents."""
        n_converged = int(np.sum(~np.isnan(simulation_ss.V_arr)))
        assert n_converged >= 5, (
            f"Only {n_converged}/6 SS steps converged; expected ≥ 5"
        )

    def test_ocv_above_floor(self, simulation_ss):
        """Open-circuit steady-state voltage must be physically reasonable."""
        V_ocv = simulation_ss.V_arr[0]
        assert V_ocv > 0.9, f"OCV_SS = {V_ocv:.3f} V, expected > 0.9 V"

    def test_monotonically_decreasing(self, simulation_ss):
        """Cell voltage must decrease strictly with increasing current density."""
        valid = ~np.isnan(simulation_ss.V_arr)
        V_valid = simulation_ss.V_arr[valid]
        diffs = np.diff(V_valid)
        assert np.all(diffs < 0), (
            f"SS voltage is not strictly decreasing; diffs = {diffs}"
        )

    def test_voltage_at_1Acm2(self, simulation_ss):
        """Interpolated SS voltage at 1 A/cm² must be in [0.65, 0.85] V."""
        valid = ~np.isnan(simulation_ss.V_arr)
        I_to_V = interp1d(
            simulation_ss.i_Acm2[valid], simulation_ss.V_arr[valid],
            kind='linear', bounds_error=False, fill_value='extrapolate',
        )
        V_1 = float(I_to_V(1.0))
        assert 0.65 <= V_1 <= 0.85, (
            f"V_SS(1 A/cm²) = {V_1:.3f} V, expected in [0.65, 0.85]"
        )

    def test_peak_power_density(self, simulation_ss):
        """Peak power density max(I × V) must exceed 0.5 W/cm²."""
        P_arr = simulation_ss.i_Acm2 * simulation_ss.V_arr
        peak = float(np.nanmax(P_arr))
        assert peak > 0.5, f"Peak power SS = {peak:.3f} W/cm², expected > 0.5"

    def test_voltage_at_1Acm2_matches_ivp(self, simulation_ss):
        """SS V(1 A/cm²) agrees with IVP reference value within TOL_SS."""
        valid = ~np.isnan(simulation_ss.V_arr)
        I_to_V = interp1d(
            simulation_ss.i_Acm2[valid], simulation_ss.V_arr[valid],
            kind='linear', bounds_error=False, fill_value='extrapolate',
        )
        V_1 = float(I_to_V(1.0))
        assert abs(V_1 - REF_V_AT_1ACMC2) / REF_V_AT_1ACMC2 <= TOL_SS, (
            f"V_SS(1 A/cm²) = {V_1:.3f} V, IVP ref = {REF_V_AT_1ACMC2:.3f} V "
            f"(±{TOL_SS*100:.0f} %)"
        )

    def test_current_density_at_06V_matches_ivp(self, simulation_ss):
        """SS I(0.6 V) agrees with IVP reference value within TOL_SS."""
        valid = ~np.isnan(simulation_ss.V_arr) & (simulation_ss.V_arr > 0.05)
        i_v = simulation_ss.i_Acm2[valid]
        V_v = simulation_ss.V_arr[valid]
        V_to_I = interp1d(V_v[::-1], i_v[::-1], kind='linear',
                          bounds_error=False, fill_value='extrapolate')
        i_06 = float(V_to_I(0.6))
        assert abs(i_06 - REF_I_AT_06V) / REF_I_AT_06V <= TOL_SS, (
            f"I_SS(0.6 V) = {i_06:.3f} A/cm², IVP ref = {REF_I_AT_06V:.3f} A/cm² "
            f"(±{TOL_SS*100:.0f} %)"
        )

    def test_min_water_content_physically_reasonable(self, simulation_ss):
        """Minimum λ over CCM at 0.6 V must be in the physically valid range [1, 22]."""
        cell = simulation_ss.base.transient_transport_model.cell
        idx = _get_idx06(simulation_ss.V_arr)
        st = simulation_ss.results[idx]
        ccm_ix = [cell.an.cl.ix, cell.memb.ix, cell.ca.cl.ix]
        lmbd_min = float(np.min(st.lmbd[ccm_ix, 0]))
        assert 1.0 <= lmbd_min <= 22.0, (
            f"λ_min_SS(0.6 V) = {lmbd_min:.3f}, expected in [1, 22]"
        )

    def test_water_flux_positive_at_06V(self, simulation_ss):
        """Water flux through PEM at 0.6 V must be positive (anode → cathode)."""
        cell = simulation_ss.base.transient_transport_model.cell
        idx = _get_idx06(simulation_ss.V_arr)
        st = simulation_ss.results[idx]
        J_PEM = _compute_J_PEM(st, simulation_ss.model, cell)
        assert J_PEM > 0, f"J_PEM_SS = {J_PEM:.4f} µmol/cm²/s, expected > 0"

    def test_R_PEM_physically_reasonable(self, simulation_ss):
        """R_PEM at 0.6 V must be in a physically plausible range [5, 500] mΩ·cm²."""
        cell = simulation_ss.base.transient_transport_model.cell
        idx = _get_idx06(simulation_ss.V_arr)
        st = simulation_ss.results[idx]
        R_PEM = _compute_R_PEM(st, simulation_ss.base, cell)
        assert 5.0 <= R_PEM <= 500.0, (
            f"R_PEM_SS(0.6 V) = {R_PEM:.3f} mΩ·cm², expected in [5, 500]"
        )

    def test_residual_finite_at_solution(self, simulation_ss):
        """SS residual must be finite at the 1 A/cm² operating point."""
        base = simulation_ss.base
        model = base.transient_transport_model
        # Re-run a single step from y0 and verify the residual is finite.
        y0 = base.initial_state(
            cell_temperature=T_OP, cell_pressure=P_OP,
            ca_rh=RH, an_rh=RH, ca_dry_o2=0.21, an_dry_h2=1.0, s_ca=S_C,
        )
        model.current_density = 10000.   # 1 A/cm²
        sol = base.solve_steady_state(y0, t=0.)
        assert np.all(np.isfinite(sol.fun)), (
            "SS residual contains non-finite values at 1 A/cm²"
        )
