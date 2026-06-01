"""
Tests for the transient cell model (new Cell / TransientCellModel API).

Covers: shape and finiteness of rates_of_change, physical plausibility of
cell voltage, and monotonicity of the polarization curve produced by
stepping the current density.
"""
import pytest
import numpy as np
from scipy.integrate import solve_ivp

import marapendi as mrpd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cl(thickness=10e-6):
    return mrpd.PtCCatalystLayer(
        thickness=thickness,
        bulk_density=2010.,
        bulk_specific_heat_capacity=710.,
        bulk_thermal_conductivity=0.25,
        L_Pt=0.3e-2,
        wt_Pt=0.4,
        ic_ratio=0.7,
        ecsa=45e3,
        ionomer=mrpd.Nafion_N21X,
        r_C=25e-9,
        K_abs=1e-13,
        theta_contact=95,
        reaction=mrpd.ElectrochemicalReaction(
            reference_exchange_current_density=2.47e-8 * 10e-6,
            activation_energy=67e6,
            reaction_order=0.54,
            reference_activity=1.,
            reference_temperature=353.15,
            number_of_electrons=2,
            charge_transfer_coeff=1,
        ),
    )


def _gdl():
    return mrpd.PorousLayer(
        thickness=160e-6,
        eps_p=0.72,
        bulk_density=440.,
        bulk_specific_heat_capacity=710.,
        bulk_thermal_conductivity=1.24,
        K_abs=1e-12,
        theta_contact=115.,
        tort=3,
    )


@pytest.fixture
def model():
    """TransientCellModel with a Nafion N212 membrane and Pt/C CLs."""
    cell = mrpd.Cell(
        area=25e-4,
        electrical_resistance=30e-7,
        thermal_resistance=2e-4,
        memb_model=mrpd.PFSAModel(),
        cl_model=mrpd.PtCCatalystLayerModel(),
        ca=mrpd.CellSide(
            cl=_cl(),
            gdl=_gdl(),
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        an=mrpd.CellSide(
            cl=_cl(),
            gdl=_gdl(),
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        memb=mrpd.Nafion_N212,
    )
    return mrpd.TransientCellModel(cell=cell)


def extract_voltage(model, x_flat, i_density):
    """Compute V_cell for a flat normalised state vector."""
    x_state = (x_flat.reshape(model.n_layers, model.n_variables, 1)
               * model.norm_factor[..., np.newaxis])
    state = model._compute_derived_quantities(x_state, i_density)
    model._compute_voltage(state)
    return float(state.V_cell[0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rates_of_change_shape_and_finite(model):
    """rates_of_change returns a finite array of the expected shape."""
    y0   = model.initial_state(T=353.15, p=1.5e5, rh=0.7)
    dxdt = model.rates_of_change(y0[:, np.newaxis], i=5000.)
    assert dxdt.shape == (model.n_layers * model.n_variables, 1)
    assert np.all(np.isfinite(dxdt))


def test_voltage_physical_range(model):
    """V_cell at moderate current density lies in the expected range [0.3, 1.1] V."""
    y0 = model.initial_state(T=353.15, p=1.5e5, rh=0.7)
    V  = extract_voltage(model, y0, i_density=5000.)
    assert 0.3 < V < 1.1, f"V_cell = {V:.3f} V is outside the expected range"


def test_polarization_curve_monotone(model):
    """V_cell decreases monotonically as current density increases."""
    current_densities = [10., 100., 500., 1000., 5000., 10000., 20000.]   # A/m²
    y = model.initial_state(T=353.15, p=1.5e5, rh=0.7)
    voltages = []

    for i_density in current_densities:
        sol = solve_ivp(
            lambda t, x: model.rates_of_change(x[:, np.newaxis], i=i_density)[:, 0],
            t_span=(0., 30.),
            y0=y,
            method='BDF',
            max_step=5.,
            rtol=1e-3,
            atol=1e-6,
        )
        y = sol.y[:, -1]
        voltages.append(extract_voltage(model, y, i_density))

    assert all(v_hi < v_lo for v_lo, v_hi in zip(voltages, voltages[1:])), (
        f"Polarization curve not monotone: {voltages}"
    )
