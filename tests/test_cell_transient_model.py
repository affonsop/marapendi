import pytest
import numpy as np
import cantera as ct
import pandas as pd
from scipy.interpolate import interp1d

from scipy.integrate import solve_ivp
import marapendi as mrpd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toray_gdl_060():
    lmbd = 0.86  # Data for figure 9 in Baker et al. (2009)
    f = 1 + 0.803 * np.exp(-1.17 * lmbd) + 0.197 * np.exp(-0.164 * lmbd)
    gdl = mrpd.PorousLayer(
        thickness=160e-6,
        eps_p=0.72,
        tort=3.,
        K_abs=1e-12,
        bulk_thermal_conductivity=1.24,
        theta_contact=115.,
    )
    return gdl


@pytest.fixture
def cl():
    return mrpd.PtCCatalystLayer(
        thickness=10e-6,
        L_Pt=0.3e-2,
        ic_ratio=0.7,
        wt_Pt=0.4,
        bulk_thermal_conductivity=0.25,
        ecsa=45e3,
        ionomer=mrpd.Nafion_N21X,
        r_C=25e-9,
        K_abs=1e-13,
        theta_contact=95,
        reaction=mrpd.ElectrochemicalReaction(
            reference_exchange_current_density=2.47e-8,
            activation_energy=67e6,
            reaction_order=0.54,
            reference_activity=1.,
            reference_temperature=353.15,
            number_of_electrons=2,
            charge_transfer_coeff=1,
        ),
    )


@pytest.fixture
def cell(cl, toray_gdl_060):
    base_cell = mrpd.Cell(
        area=25e-4,
        electrical_resistance=30e-7,
        thermal_resistance=2e-4,
        memb_model=mrpd.PFSAModel(),
        cl_model=mrpd.PtCCatalystLayerModel(),
        ca=mrpd.CellSide(
            cl=mrpd.PtCCatalystLayer(
                thickness=10e-6,
                bulk_density=2010.,
                bulk_specific_heat_capacity=710.,
                L_Pt=0.3e-2,
                ic_ratio=0.7,
                wt_Pt=0.4,
                bulk_thermal_conductivity=0.25,
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
            ),
            gdl=mrpd.PorousLayer(
                thickness=160e-6,
                eps_p=0.72,
                bulk_density=440.,
                bulk_specific_heat_capacity=710.,
                K_abs=1e-12,
                bulk_thermal_conductivity=1.24,
                theta_contact=115.,
                tort=3,
            ),
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        an=mrpd.CellSide(
            cl=mrpd.PtCCatalystLayer(
                thickness=10e-6,
                bulk_density=2010.,
                bulk_specific_heat_capacity=710.,
                L_Pt=0.3e-2,
                ic_ratio=0.7,
                wt_Pt=0.4,
                bulk_thermal_conductivity=0.25,
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
            ),
            gdl=mrpd.PorousLayer(
                thickness=160e-6,
                K_abs=1e-12,
                bulk_thermal_conductivity=1.24,
                theta_contact=115.,
                tort=3,
            ),
            ch=mrpd.FlowChannel(height=1e-3, bulk_thermal_conductivity=100.),
            has_mpl=False,
        ),
        memb=mrpd.Nafion_N212,
    )
    return mrpd.TransientCellModel(cell=base_cell)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

df = pd.read_csv('data/test_ui_curve.csv')


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test(cell):
    # Convenience aliases
    c = cell.cell

    i = interp1d(
        df['t_step'].values,
        1e4 * np.maximum(0, df['I_step']).values,
        fill_value=0,
        bounds_error=False,
    )

    x0 = np.array(
        [
            [
                [14]                              * cell.n_layers,
                [353.15]                          * cell.n_layers,
                [1e5 / ct.gas_constant / 353.15 * .4]  * cell.n_layers,
                [1e5 / ct.gas_constant / 353.15 * .2]  * cell.n_layers,
                [1e5 / ct.gas_constant / 353.15 * .0]  * cell.n_layers,
                [1e5 / ct.gas_constant / 353.15 * 0.4] * cell.n_layers,
                [0.1]                             * cell.n_layers,
            ],
            [
                [10]                              * cell.n_layers,
                [343.15]                          * cell.n_layers,
                [1e5 / ct.gas_constant / 353.15 * .3]  * cell.n_layers,
                [1e5 / ct.gas_constant / 353.15 * .2]  * cell.n_layers,
                [1e5 / ct.gas_constant / 353.15 * .2]  * cell.n_layers,
                [1e5 / ct.gas_constant / 353.15 * 0.3] * cell.n_layers,
                [0.1]                             * cell.n_layers,
            ],
        ]
    ).transpose()

    def f(t, x):
        return cell.rates_of_change(x, i=i(t))

    import time
    t1 = time.time()
    tf = df['t_step'].values[-1]
    current_density = i(df['t_step'])

    sol = solve_ivp(
        f,
        t_span=(0, tf),
        t_eval=df['t_step'].values,
        y0=(x0[..., 1] / cell.norm_factor).reshape(cell.n_layers * cell.n_variables),
        method='BDF',
        vectorized=True,
        max_step=10,
    )

    state = cell._compute_derived_quantities(
        sol.y.reshape(cell.n_layers, cell.n_variables, sol.y.shape[-1])
        * cell.norm_factor[..., np.newaxis],
        current_density,
    )

    t2 = time.time()
    print(sol, tf / (t2 - t1))

    plt.figure()
    plt.plot(sol.t, state.lmbd[2, ...])
    plt.plot(sol.t, state.lmbd[3, ...])
    plt.plot(sol.t, state.lmbd[4, ...])
    ax2 = plt.gca().twinx()
    ax2.plot(df['t_step'], i(df['t_step']))

    plt.figure()
    plt.plot(sol.t, state.s[c.ca.cl.ix, ...])
    plt.plot(sol.t, state.s[c.ca.gdl.ix, ...])

    plt.figure()
    plt.plot(sol.t, state.T[c.ca.ch.ix, ...])
    plt.plot(sol.t, state.T[c.ca.cl.ix, ...])
    plt.plot(sol.t, state.T[c.memb.ix, ...])
    plt.plot(sol.t, state.T[c.an.cl.ix, ...])
    plt.plot(sol.t, state.T[c.an.ch.ix, ...])

    plt.figure()
    for k in [3]:
        plt.plot(sol.t, state.cg_k[c.ca.cl.ix, k, ...]  / mrpd.water_saturation_concentration(state.T_ca_cl), label='ca cl')
        plt.plot(sol.t, state.cg_k[c.ca.gdl.ix, k, ...] / mrpd.water_saturation_concentration(state.T_ca_cl), label='ca gdl')
        plt.plot(sol.t, state.cg_k[c.an.cl.ix, k, ...]  / mrpd.water_saturation_concentration(state.T_ca_cl), label='an cl')
        plt.legend()

    plt.figure()
    plt.plot(current_density, state.V_cell, '-')

    plt.show()
    assert False
