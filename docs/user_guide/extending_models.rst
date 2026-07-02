Extending models by inheritance
================================

**marapendi** is designed around composable strategy objects:
:class:`~marapendi.cell.explicit_steady_state.ExplicitSteadyStateModel` owns a
``voltage_model``, a ``thermal_model``, a ``water_balance_model``, and a
``gas_transport_model``.  Any of these can be replaced by a subclass.  The cell
solver itself can also be subclassed to add post-processing steps or to change
the iteration logic.

The reference for inheritance patterns is
:class:`~marapendi.cell.implicit_steady_state.ImplicitSteadyStateModel`, which
extends :class:`~marapendi.cell.explicit_steady_state.ExplicitSteadyStateModel`
by overriding only ``solve`` and leaving everything else unchanged.

Pattern 1 — custom ionomer isotherm
--------------------------------------

Override :meth:`~marapendi.membrane.pem.PFSAIonomer.vapor_equilibrium_water_content`
to supply a different λ(RH) relationship, for example a temperature-dependent
polynomial fit:

.. code-block:: python

    from dataclasses import dataclass
    import numpy as np
    import marapendi as mrpd

    @dataclass
    class MyIonomer(mrpd.PFSAIonomer):
        """PFSA ionomer with a custom temperature-dependent isotherm."""

        def vapor_equilibrium_water_content(self, rh, temperature):
            """Example: Hinatsu et al. (1994) temperature-dependent polynomial."""
            a = 0.043 + 17.81 * rh - 39.85 * rh**2 + 36.0 * rh**3
            # Temperature correction term
            b = 0.768 + 0.664 * rh
            return a + b * (temperature - 303.15) / 30.0

    ionomer = MyIonomer(equivalent_weight=1100, dry_density=1980)
    # Pass ionomer to PtCCatalystLayer and PFSA as usual
    # membrane = mrpd.PFSA(ionomer=ionomer, dry_thickness=25e-6)

.. note::

   The base class
   :meth:`~marapendi.membrane.pem.PFSAIonomer.fit_rh_piecewise_linear` calls
   ``vapor_equilibrium_water_content`` to fit the PWL approximation used by
   :class:`~marapendi.water_balance.membrane_pwl.MembraneWaterBalanceModelPiecewise`.
   If you subclass ``vapor_equilibrium_water_content``, the PWL fit is
   re-computed automatically.

Pattern 2 — custom kinetics (VoltageModel)
-------------------------------------------

:class:`~marapendi.cell.voltage.VoltageModel` computes the activation
overpotential, ohmic overpotential, and cell voltage.  Subclass it to change the
ORR kinetic expression while keeping the rest of the voltage model intact:

.. code-block:: python

    from dataclasses import dataclass
    from marapendi.cell.voltage import VoltageModel
    from marapendi.thermo.electrochemistry import ElectrochemicalReaction
    from marapendi.cell.explicit_steady_state import ExplicitSteadyStateModel
    import numpy as np

    @dataclass
    class TafelVoltageModel(VoltageModel):
        """Replace the built-in ORR expression with a simple Tafel law."""

        def activation_overpotential(self, cell, state, theta_PtO=0.0):
            i_loc = state.current_density + state.crossover_current
            i0    = cell.ca.cl.reaction.reference_exchange_current_density
            b     = 0.5                # charge-transfer coefficient
            F     = 96485.
            R     = 8.314
            T     = state.ca.cl.temperature
            # Butler-Volmer / Tafel approximation
            return (R * T) / (b * F) * np.log(np.maximum(i_loc / i0, 1e-30))

    model = ExplicitSteadyStateModel(
        voltage_model=TafelVoltageModel()
    )

Pattern 3 — post-processing step in a model subclass
------------------------------------------------------

The cleanest way to add derived outputs (for example, a custom degradation
indicator) without touching the library source is to wrap ``solve``:

.. code-block:: python

    from dataclasses import dataclass
    from marapendi.cell.explicit_steady_state import ExplicitSteadyStateModel

    @dataclass
    class DiagnosticModel(ExplicitSteadyStateModel):
        """Explicit SS model that also computes HFR and stores it in state."""

        def solve(self, cell, cell_conditions, initial_state):
            state = super().solve(cell, cell_conditions, initial_state)
            # Attach HFR so downstream code can use state.hfr directly
            state.hfr = self.voltage_model.high_frequency_resistance(cell, state)
            return state

    model = DiagnosticModel()
    state = model.solve(cell, conditions, model.set_initial_conditions(cell, conditions))
    print(state.hfr)   # now populated

Pattern 4 — custom thermal model
----------------------------------

:class:`~marapendi.cell.thermal.ThermalModel` computes the MEA temperature from
the thermal resistance and heat release.  Subclass it to change the boundary
conditions — for example, to include convective cooling by the coolant channel:

.. code-block:: python

    from dataclasses import dataclass
    from marapendi.cell.thermal import ThermalModel

    @dataclass
    class CoolantThermalModel(ThermalModel):
        """Adds a convective term between the MEA and a coolant channel."""

        coolant_temperature: float = 348.15   # K
        coolant_htc: float = 500.             # W / (m² K)

        def heat_transfer_resistance(self, cell):
            R_base = super().heat_transfer_resistance(cell)
            R_coolant = 1.0 / (self.coolant_htc * cell.area)
            return R_base + R_coolant

        def mea_temperature(self, cell, state, cell_voltage):
            heat_gen = state.heat_release
            T_mea = (
                self.coolant_temperature
                + heat_gen * self.heat_transfer_resistance(cell)
            )
            return T_mea

    model = mrpd.ExplicitSteadyStateModel(
        thermal_model=CoolantThermalModel(coolant_temperature=345.)
    )

Pattern 5 — subclassing ImplicitSteadyStateModel
--------------------------------------------------

:class:`~marapendi.cell.implicit_steady_state.ImplicitSteadyStateModel` is
itself a subclass of
:class:`~marapendi.cell.explicit_steady_state.ExplicitSteadyStateModel` and
overrides only ``solve``.  You can further subclass it while keeping the implicit
temperature–voltage iteration:

.. code-block:: python

    from dataclasses import dataclass
    from marapendi.cell.implicit_steady_state import ImplicitSteadyStateModel

    @dataclass
    class ImplicitWithHFR(ImplicitSteadyStateModel):
        """Implicit model that also stores HFR in state."""

        def solve(self, cell, cell_conditions, initial_state):
            state = super().solve(cell, cell_conditions, initial_state)
            state.hfr = self.voltage_model.high_frequency_resistance(cell, state)
            return state
