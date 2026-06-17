"""Integration tests for FuelCell polarization-curve simulation.

Uses a simplified cell (no MPL, default Nafion membrane) with typical
operating conditions at 353 K to verify the simulation pipeline runs
without error and produces physically plausible results.
"""
import numpy as np
import pytest
import marapendi as mrpd


def _make_cell():
    liq_model = mrpd.DarcyTransportModel(J_function_exponent=2)

    gdl = mrpd.GasDiffusionLayer(
        thickness=200e-6,
        porosity=0.6,
        contact_angle=120.,
        effective_gas_diffusion_ratio=0.3,
        absolute_permeability=1e-12,
        thermal_conductivity=0.5,
        two_phase_transport_model=liq_model,
    )

    cl_ca = mrpd.PtCCatalystLayer(
        ecsa=70e3,
        platinum_loading=0.4e-2,
        ionomer=mrpd.PFSAIonomer(),
        reaction=mrpd.ElectrochemicalReaction(
            reference_exchange_current_density=2.5e-4,
            reaction_order=0.54,
            activation_energy=67e6,
            reference_activity=1e5,
            reference_temperature=353.15,
            number_of_electrons=2,
            charge_transfer_coeff=0.5,
        ),
        thickness=10e-6,
        thermal_conductivity=0.22,
        pore_diameter=40e-9,
        absolute_permeability=1e-13,
        contact_angle=97.,
        two_phase_transport_model=liq_model,
    )

    ch = mrpd.FlowChannel(
        width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='o2',
    )

    return mrpd.FuelCell(
        area=25e-4,
        electrical_resistance=30e-7,
        ca=mrpd.FuelCellSide(
            cl=cl_ca,
            gdl=mrpd.GasDiffusionLayer(
                thickness=200e-6,
                effective_gas_diffusion_ratio=0.3,
                thermal_conductivity=0.5,
                two_phase_transport_model=liq_model,
            ),
            ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1,
                                 n_parallel=20, reactant='o2'),
            has_mpl=False,
            thermal_contact_resistance=4e-4,
        ),
        an=mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(
                thickness=5e-6,
                two_phase_transport_model=liq_model,
            ),
            gdl=gdl,
            ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1,
                                  n_parallel=20, reactant='h2'),
            has_mpl=False,
            thermal_contact_resistance=4e-4,
        ),
        membrane=mrpd.PFSA(
            equivalent_weight=1100,
            dry_density=1980,
            dry_thickness=25e-6,
            water_balance_model=mrpd.MembraneWaterBalanceModel(),
        ),
        use_eq_water_content_for_ionomer=True,
    )


def _cathode_conditions(T=353.15, rh=0.5, p=1.5e5, st=2.0):
    return mrpd.OperatingConditions(
        inlet_temperature=T,
        inlet_pressure=p,
        outlet_pressure=p,
        dry_o2_mole_fraction=0.21,
        inlet_relative_humidity=rh,
        stoichiometry=st,
    )


def _anode_conditions(T=353.15, rh=0.5, p=1.5e5, st=1.5):
    return mrpd.OperatingConditions(
        inlet_temperature=T,
        inlet_pressure=p,
        outlet_pressure=p,
        dry_h2_mole_fraction=1.0,
        inlet_relative_humidity=rh,
        stoichiometry=st,
    )


@pytest.fixture
def fuel_cell():
    return _make_cell()


class TestFuelCellConstruction:
    def test_ca_reactant(self, fuel_cell):
        assert fuel_cell.ca.reactant == 'o2'

    def test_an_reactant(self, fuel_cell):
        assert fuel_cell.an.reactant == 'h2'

    def test_porous_layers_list(self, fuel_cell):
        # Without MPL: gdl + cl
        assert len(fuel_cell.ca.porous_layers) == 2

    def test_membrane_instantiated(self, fuel_cell):
        assert isinstance(fuel_cell.membrane, mrpd.PFSA)


class TestPolarizationCurve:
    def test_voltage_positive_at_low_current(self, fuel_cell):
        T = 353.15
        i = np.array([1e3])   # 0.1 A/cm²
        V = fuel_cell.compute_ui_curve(i, T, _cathode_conditions(T), _anode_conditions(T))
        V = np.atleast_1d(V)
        assert float(V[0]) > 0.5

    def test_voltage_below_open_circuit(self, fuel_cell):
        T = 353.15
        i = np.array([1e3])
        V = fuel_cell.compute_ui_curve(i, T, _cathode_conditions(T), _anode_conditions(T))
        V = np.atleast_1d(V)
        assert float(V[0]) < 1.23

    def test_voltage_decreases_with_current(self, fuel_cell):
        T = 353.15
        i_vals = np.array([1e3, 5e3, 1e4])
        voltages = []
        for i in i_vals:
            V = fuel_cell.compute_ui_curve(np.array([i]), T, _cathode_conditions(T), _anode_conditions(T))
            voltages.append(float(np.atleast_1d(V)[0]))
        assert voltages[0] > voltages[1] > voltages[2]

    def test_hfr_positive(self, fuel_cell):
        T = 353.15
        i = np.array([5e3])
        fuel_cell.compute_ui_curve(i, T, _cathode_conditions(T), _anode_conditions(T))
        hfr = fuel_cell.high_frequency_resistance()
        assert float(np.atleast_1d(hfr)[0]) > 0

    def test_voltage_higher_at_elevated_pressure(self, fuel_cell):
        T = 353.15
        i = np.array([5e3])
        V_low = float(np.atleast_1d(fuel_cell.compute_ui_curve(
            i, T, _cathode_conditions(T, p=1.5e5), _anode_conditions(T, p=1.5e5),
        ))[0])
        V_high = float(np.atleast_1d(fuel_cell.compute_ui_curve(
            i, T, _cathode_conditions(T, p=2.5e5), _anode_conditions(T, p=2.5e5),
        ))[0])
        assert V_high > V_low

    def test_vectorised_sweep(self, fuel_cell):
        T = 353.15
        i_arr = np.array([1e3, 5e3, 1e4, 2e4])
        V = fuel_cell.compute_ui_curve(i_arr, T, _cathode_conditions(T), _anode_conditions(T))
        V = np.atleast_1d(V)
        assert V.shape == i_arr.shape
        assert np.all(np.diff(V) < 0)  # monotonically decreasing
