"""
Tests verifying that mutating a Cell in-place (the parameter-estimation
closure pattern) produces identical results to building a fresh Cell with
the same parameter values.

The canonical update pattern is:
    cell.electrical_resistance = new_value
    cell.ca.cl.reaction.reference_exchange_current_density = new_value
    cell.build_property_arrays()   # always rebuild cached arrays

`build_property_arrays` is called unconditionally so that PorousLayer /
Membrane parameters (thickness, K_abs, porosity, …) — which feed into the
cached arrays Cell.thickness, Cell.K_abs, etc. — are always kept consistent.
Forgetting the call would silently produce wrong results for that class of
parameters; calling it when only Cell-level or ElectrochemicalReaction scalars
changed is harmless and cheap.
"""
import numpy as np
import pytest
import marapendi.dynamic as mrpd

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

T_OP = 343.15
P_OP = 1.5e5
RH   = 0.9
S_C  = 0.12

# A handful of moderate current densities — fast to solve, wide enough to
# exercise both the activation and ohmic regions.
CURRENT_DENSITIES = np.array([500., 5000., 10000., 14000.])   # A/m²


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nafion():
    return mrpd.PFSAIonomer(
        rho_dry_ion=1.97e3, EW_ion=1020,
        darken_num_ion=np.array([0., 67.74, -32.03, 3.842]),
        darken_den_ion=np.array([103.37, -33.013, -2.115, 1.0]),
        sorption_coeffs_ion=np.array([0.043, 17.81, -39.85, 36.0]),
        lmbd_liq_ref_ion=22, D_lmbd_ref_ion=1.0e-10, k_des_ref_ion=1.42e-4,
        E_act_ion=20e6, E_act_cond_ion=15e6, sigma_ref_ion=116.,
        f_v_perc_ion=0.06, n_sigma_ion=1.5,
        T_ref_sigma_ion=353.15, T_ref_D_ion=353.15, T_ref_des_ion=353.15,
    )


def _gdl_kw():
    return dict(thickness=160e-6, eps_p=0.76, bulk_density=440.,
                bulk_specific_heat_capacity=710., bulk_thermal_conductivity=1.6,
                K_abs=6.15e-12, theta_contact=130., tort=1.6**2)


def _make_cell(r_elec: float, i0: float):
    """Build a fresh Cell with the given electrical resistance and i0."""
    nafion = _make_nafion()
    orr = mrpd.ElectrochemicalReaction(
        reference_exchange_current_density=i0,
        activation_energy=67e6, reaction_order=0.54,
        reference_activity=1.e5, reference_temperature=353.15,
        number_of_electrons=1, charge_transfer_coeff=1,
    )
    gdl_kw = _gdl_kw()
    cl_kw = dict(thickness=10e-6, bulk_density=1000., bulk_specific_heat_capacity=710.,
                 bulk_thermal_conductivity=0.27, L_Pt=0.4e-2, wt_Pt=0.416,
                 ic_ratio=1.04, ecsa=75e3, tort=1.6**2, ionomer=nafion,
                 r_C=1e-10, K_abs=1e-13, theta_contact=95)
    an_cl_kw = dict(thickness=10e-6, bulk_density=1000., bulk_specific_heat_capacity=710.,
                    bulk_thermal_conductivity=0.27, L_Pt=0.4e-2/3, wt_Pt=0.192,
                    ic_ratio=1.07, ecsa=75e3, tort=1.6**2, ionomer=nafion,
                    r_C=1e-10, K_abs=1e-13, theta_contact=95)
    return mrpd.Cell(
        area=25e-4, electrical_resistance=r_elec, thermal_resistance=0,
        ca=mrpd.CellSide(
            cl=mrpd.PtCCatalystLayer(**{**cl_kw, 'reaction': orr}),
            gdl=mrpd.PorousLayer(**gdl_kw),
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        an=mrpd.CellSide(
            cl=mrpd.PtCCatalystLayer(**an_cl_kw),
            gdl=mrpd.PorousLayer(**gdl_kw),
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        memb=mrpd.PFSAMembrane(thickness=25e-6, bulk_thermal_conductivity=0.3,
                                ionomer=nafion),
    )


def _make_base(cell):
    return mrpd.CellBaseModel(
        transient_transport_model=mrpd.TransientCellModel(cell=cell, current_density=0.),
        memb_model=mrpd.PFSAModel(),
        cl_model=mrpd.PtCCatalystLayerModel(),
        gas_diffusion_model=mrpd.PorousGasResistanceModel(),
        darcy_transport_model=mrpd.DarcyTransportModel(),
        voltage_model=mrpd.VoltageModel(),
    )


def _sweep(base, y0, *, rebuild_arrays: bool = True) -> np.ndarray:
    """Steady-state sweep; returns V_cell at CURRENT_DENSITIES (NaN on failure).

    Parameters
    ----------
    rebuild_arrays : bool
        If True (default), call ``cell.build_property_arrays()`` before the
        sweep — mirrors the recommended pattern in model_fn.  Pass False only
        to test what happens when the rebuild is skipped.
    """
    model = base.transient_transport_model
    if rebuild_arrays:
        model.cell.build_property_arrays()
    V = np.full(len(CURRENT_DENSITIES), np.nan)
    y = y0.copy()
    for k, i_k in enumerate(CURRENT_DENSITIES):
        model.current_density = float(i_k)
        sol = base.solve_steady_state(y, t=0.)
        if not sol.success:
            break
        st = base.postprocess(sol.y, i_density=float(i_k))
        V_k = float(st.V_cell[0])
        if V_k <= 0.:
            break
        V[k] = V_k
        y = sol.y[:, 0]
    return V


def _y0(base):
    return base.initial_state(
        cell_temperature=T_OP, cell_pressure=P_OP,
        ca_rh=RH, an_rh=RH, ca_dry_o2=0.21, an_dry_h2=1.0, s_ca=S_C,
    )


# ---------------------------------------------------------------------------
# Parameters used in the tests
# ---------------------------------------------------------------------------

R_A, R_B = 9.4e-5, 5.0e-5     # two distinct electrical resistances [Ω·m²]
I0_A, I0_B = 2.45e-4, 1.0e-4  # two distinct exchange current densities [A/m²_Pt]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestCellUpdateConsistency:
    """In-place cell mutation must be transparent to the steady-state solver."""

    def test_update_electrical_resistance(self):
        """Sweeping after cell.electrical_resistance = R_B equals a fresh build."""
        # Fresh build at R_B
        cell_b     = _make_cell(r_elec=R_B, i0=I0_A)
        base_b     = _make_base(cell_b)
        V_fresh    = _sweep(base_b, _y0(base_b))

        # Start at R_A, then update in-place to R_B
        cell_mut   = _make_cell(r_elec=R_A, i0=I0_A)
        base_mut   = _make_base(cell_mut)
        y0_mut     = _y0(base_mut)
        cell_mut.electrical_resistance = R_B
        V_updated  = _sweep(base_mut, y0_mut)

        np.testing.assert_allclose(
            V_updated, V_fresh, rtol=1e-10,
            err_msg="Voltage after in-place r_elec update differs from fresh build",
        )

    def test_update_exchange_current_density(self):
        """Sweeping after reaction.reference_exchange_current_density = I0_B equals a fresh build."""
        cell_b  = _make_cell(r_elec=R_A, i0=I0_B)
        base_b  = _make_base(cell_b)
        V_fresh = _sweep(base_b, _y0(base_b))

        cell_mut = _make_cell(r_elec=R_A, i0=I0_A)
        base_mut = _make_base(cell_mut)
        y0_mut   = _y0(base_mut)
        cell_mut.ca.cl.reaction.reference_exchange_current_density = I0_B
        V_updated = _sweep(base_mut, y0_mut)

        np.testing.assert_allclose(
            V_updated, V_fresh, rtol=1e-10,
            err_msg="Voltage after in-place i0 update differs from fresh build",
        )

    def test_update_both_parameters(self):
        """Updating both parameters simultaneously equals a fresh build at (R_B, I0_B)."""
        cell_b  = _make_cell(r_elec=R_B, i0=I0_B)
        base_b  = _make_base(cell_b)
        V_fresh = _sweep(base_b, _y0(base_b))

        cell_mut = _make_cell(r_elec=R_A, i0=I0_A)
        base_mut = _make_base(cell_mut)
        y0_mut   = _y0(base_mut)
        cell_mut.electrical_resistance = R_B
        cell_mut.ca.cl.reaction.reference_exchange_current_density = I0_B
        V_updated = _sweep(base_mut, y0_mut)

        np.testing.assert_allclose(
            V_updated, V_fresh, rtol=1e-10,
            err_msg="Voltage after in-place update of both params differs from fresh build",
        )

    def test_repeated_updates_are_idempotent(self):
        """Updating to R_B twice gives the same result as updating once."""
        cell_mut = _make_cell(r_elec=R_A, i0=I0_A)
        base_mut = _make_base(cell_mut)
        y0_mut   = _y0(base_mut)

        cell_mut.electrical_resistance = R_B
        V_first  = _sweep(base_mut, y0_mut)

        cell_mut.electrical_resistance = R_B   # same value again
        V_second = _sweep(base_mut, y0_mut)

        np.testing.assert_allclose(V_first, V_second, rtol=1e-10)

    def test_build_property_arrays_required_for_porous_layer_params(self):
        """Skipping build_property_arrays after changing a PorousLayer field
        leaves the cached Cell arrays stale, producing a different (wrong) voltage.

        This test documents *why* build_property_arrays() must always be called,
        even for parameters that are not directly on Cell.electrical_resistance or
        ElectrochemicalReaction.  Here we mutate gdl.K_abs (absolute permeability),
        which feeds into Cell.K_abs via build_property_arrays.
        """
        K_abs_A = 6.15e-12   # original value
        K_abs_B = 1.00e-12   # different value — changes liquid transport

        cell_b     = _make_cell(r_elec=R_A, i0=I0_A)
        # Manually set the GDL K_abs to K_abs_B on the fresh cell
        cell_b.ca.gdl.K_abs = K_abs_B
        cell_b.an.gdl.K_abs = K_abs_B
        cell_b.build_property_arrays()   # fresh cell built correctly
        base_b     = _make_base(cell_b)
        V_fresh    = _sweep(base_b, _y0(base_b))

        # Mutation path WITH rebuild
        cell_mut   = _make_cell(r_elec=R_A, i0=I0_A)
        base_mut   = _make_base(cell_mut)
        y0_mut     = _y0(base_mut)
        cell_mut.ca.gdl.K_abs = K_abs_B
        cell_mut.an.gdl.K_abs = K_abs_B
        V_with_rebuild    = _sweep(base_mut, y0_mut, rebuild_arrays=True)
        V_without_rebuild = _sweep(base_mut, y0_mut, rebuild_arrays=False)

        # With rebuild: must match the fresh build
        np.testing.assert_allclose(
            V_with_rebuild, V_fresh, rtol=1e-10,
            err_msg="With rebuild, mutated K_abs should equal fresh build",
        )
        # Without rebuild: stale Cell.K_abs → different (wrong) result
        assert not np.allclose(V_without_rebuild, V_fresh, rtol=1e-6), (
            "Without rebuild, stale Cell.K_abs should produce a different curve"
        )

    def test_roundtrip_restores_original_curve(self):
        """A ← B ← A gives back the original voltage curve."""
        cell_mut = _make_cell(r_elec=R_A, i0=I0_A)
        base_mut = _make_base(cell_mut)
        y0_mut   = _y0(base_mut)

        V_original = _sweep(base_mut, y0_mut)

        # Change to B
        cell_mut.electrical_resistance = R_B
        cell_mut.ca.cl.reaction.reference_exchange_current_density = I0_B
        _sweep(base_mut, y0_mut)   # discard result

        # Restore to A
        cell_mut.electrical_resistance = R_A
        cell_mut.ca.cl.reaction.reference_exchange_current_density = I0_A
        V_restored = _sweep(base_mut, y0_mut)

        np.testing.assert_allclose(
            V_restored, V_original, rtol=1e-10,
            err_msg="Roundtrip A→B→A did not restore the original voltage curve",
        )
