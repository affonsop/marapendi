.. _building_a_cell:

Building a cell model
=====================

This page walks through the minimum code required to assemble a PEM fuel cell
and simulate a polarization curve.  For a runnable version, see
``notebooks/01_polarization_curve.ipynb``.

Imports
-------

.. code-block:: python

    import numpy as np
    import marapendi as mrpd

Step 1 — Assemble the component tree
-------------------------------------

The component tree is built bottom-up: inner layers first, then side objects,
then the cell.

.. code-block:: text

    FuelCell
     ├── ca : FuelCellSide
     │    ├── cl  : PtCCatalystLayer
     │    ├── gdl : GasDiffusionLayer
     │    └── ch  : FlowChannel
     ├── an : FuelCellSide
     │    └── (same structure)
     └── membrane : PFSA

Two-phase transport model
~~~~~~~~~~~~~~~~~~~~~~~~~

The Darcy two-phase transport model is shared across all porous layers:

.. code-block:: python

    liq = mrpd.DarcyTransportModel(J_function_exponent=2)

Cathode catalyst layer
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    orr = mrpd.ElectrochemicalReaction(
        reference_exchange_current_density=2.5e-4,   # A/m²_Pt
        reaction_order=0.54,
        activation_energy=67e6,                       # J/kmol
        reference_activity=1e5,                        # Pa
        reference_temperature=353.15,
        number_of_electrons=2,
        charge_transfer_coeff=0.5,
    )

    cl_ca = mrpd.PtCCatalystLayer(
        ecsa=70e3,                # m²/kg
        platinum_loading=0.4e-2, # kg/m²
        ionomer=mrpd.PFSAIonomer(),
        reaction=orr,
        thickness=10e-6,
        two_phase_transport_model=liq,
    )

Gas diffusion layer
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    gdl = mrpd.GasDiffusionLayer(
        thickness=200e-6,
        porosity=0.6,
        contact_angle=120.,
        effective_gas_diffusion_ratio=0.3,
        absolute_permeability=1e-12,
        thermal_conductivity=0.5,
        two_phase_transport_model=liq,
    )

Flow channel
~~~~~~~~~~~~

.. code-block:: python

    ch_ca = mrpd.FlowChannel(
        width=1e-3, height=1e-3, length=0.1, n_parallel=20,
        reactant='o2',
    )

Membrane
~~~~~~~~

.. code-block:: python

    membrane = mrpd.PFSA(
        equivalent_weight=1100,
        dry_density=1980,
        dry_thickness=25e-6,
        water_balance_model=mrpd.MembraneWaterBalanceModel(),
    )

Assembling the cell
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    cell = mrpd.FuelCell(
        area=25e-4,                   # m²
        electrical_resistance=30e-7,  # Ω·m²
        ca=mrpd.FuelCellSide(
            cl=cl_ca, gdl=gdl, ch=ch_ca,
            has_mpl=False, thermal_contact_resistance=4e-4,
        ),
        an=mrpd.FuelCellSide(
            cl=mrpd.PtCCatalystLayer(thickness=5e-6, two_phase_transport_model=liq),
            gdl=gdl,
            ch=mrpd.FlowChannel(width=1e-3, height=1e-3, length=0.1, n_parallel=20, reactant='h2'),
            has_mpl=False, thermal_contact_resistance=4e-4,
        ),
        membrane=membrane,
        use_eq_water_content_for_ionomer=True,
    )

Step 2 — Define operating conditions
--------------------------------------

:class:`~marapendi.simulation.state.OperatingConditions` groups all inlet
boundary conditions for one side:

.. code-block:: python

    T = 353.15  # K

    ca_cond = mrpd.OperatingConditions(
        inlet_temperature=T,
        inlet_pressure=1.5e5,
        outlet_pressure=1.5e5,
        dry_o2_mole_fraction=0.21,
        inlet_relative_humidity=0.5,
        stoichiometry=2.0,
    )

    an_cond = mrpd.OperatingConditions(
        inlet_temperature=T,
        inlet_pressure=1.5e5,
        outlet_pressure=1.5e5,
        dry_h2_mole_fraction=1.0,
        inlet_relative_humidity=0.5,
        stoichiometry=1.5,
    )

Step 3 — Simulate
-----------------

Pass a NumPy array of current densities (A/m²) to ``compute_ui_curve``:

.. code-block:: python

    i_Am2 = np.array([1e3, 5e3, 1e4, 1.5e4, 2e4])
    V = cell.compute_ui_curve(i_Am2, T, ca_cond, an_cond)

Post-simulation state
---------------------

After calling ``compute_ui_curve``, the following attributes are available on
the ``cell`` object (and on ``cell.state`` for the last operating point):

+------------------------------+-------------------------------------+
| Attribute                    | Description                         |
+==============================+=====================================+
| ``cell.cell_voltage``        | Cell voltage (V) at last point      |
+------------------------------+-------------------------------------+
| ``cell.mea_temperature``     | MEA temperature (K)                 |
+------------------------------+-------------------------------------+
| ``cell.high_frequency_resistance()`` | HFR (Ω·m²)              |
+------------------------------+-------------------------------------+
| ``cell.activation_overpotential()``  | ORR overpotential (V)    |
+------------------------------+-------------------------------------+
| ``cell.ohmic_overpotential()``       | Ohmic voltage loss (V)   |
+------------------------------+-------------------------------------+
| ``cell.state.membrane.water_content``| Mean membrane water content|
+------------------------------+-------------------------------------+

Internal solve sequence
-----------------------

``compute_ui_curve`` calls :class:`~marapendi.models.cell.explicit_steady_state.ExplicitSteadyStateModel`
which runs the following steps in order:

1. **Set initial state** — temperatures, inlet gas compositions, flow rates.
2. **Thermal model** — compute MEA heat-transfer resistance and MEA temperature.
3. **Water balance** — solve membrane water-content profile, compute net water
   flux and electrode water saturation.
4. **Gas transport** — compute gas-transport resistances and CL gas compositions.
5. **Voltage model** — compute reversible voltage, activation, ohmic and
   concentration overpotentials, then cell voltage.
