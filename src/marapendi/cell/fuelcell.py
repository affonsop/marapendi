"""
PEM fuel cell component: :class:`FuelCell` and :class:`FuelCellSide`.

:class:`FuelCell` is the component tree a user builds to describe a cell's
geometry and materials.  Physics are evaluated by creating a model object
and calling its :meth:`~ExplicitSteadyStateModel.set_initial_conditions` /
:meth:`~ExplicitSteadyStateModel.solve` pair:

::

    cell = FuelCell(
        area=25e-4,
        ca=FuelCellSide(cl=PtCCatalystLayer(...), gdl=GasDiffusionLayer(...), ch=FlowChannel(...)),
        an=FuelCellSide(...),
        membrane=PFSA(...),
    )

    model = ExplicitSteadyStateModel()
    conditions = CellConditions(
        current_density=np.linspace(1e3, 2e4, 20),
        cell_temperature=353.15,
        ca=SideConditions(outlet_pressure=1.5e5, dry_o2_mole_fraction=0.21, ...),
        an=SideConditions(outlet_pressure=1.5e5, dry_h2_mole_fraction=1.0, ...),
    )
    state = model.set_initial_conditions(cell, conditions)
    state = model.solve(cell, conditions, state)
    # state.cell_voltage, state.mea_temperature, … are now populated
"""
from dataclasses import dataclass, field
import numpy as np
from ..porous_layers.porous_layers import PorousLayer
from ..porous_layers.catalyst_layers import PtCCatalystLayer
from ..channel.flow_channels import FlowChannel
from ..membrane.membrane_base import Membrane
from .cell import Cell, CellSide
from .voltage import VoltageModel
from .thermal import ThermalModel
from ..electrolyte.electrolyte import ElectrolyteSolution
from .explicit_steady_state import ExplicitSteadyStateModel
from .implicit_steady_state import ImplicitSteadyStateModel
from .state import (
    CellState, CellSideState, LayerState, CatalystLayerState,
    FlowChannelState, MembraneState,
)
from .state import GasState


@dataclass
class FuelCellSide(CellSide):
    """
    One side (anode or cathode) of a PEM fuel cell.

    Owns the porous layers (catalyst layer, optional microporous layer, gas
    diffusion layer) and flow channel for one electrode.

    Attributes
    ----------
    cl : CatalystLayer
        Catalyst layer (defaults to :class:`~marapendi.PtCCatalystLayer`).
    gdl : PorousLayer
        Gas diffusion layer.
    mpl : PorousLayer, optional
        Microporous layer.
    ch : FlowChannel
        Flow channel.
    has_mpl : bool
        Whether this side includes a microporous layer.
    is_wet : bool
        Whether this side is flooded with liquid (used for water electrolysis).
    thermal_contact_resistance : float
        Additional thermal contact resistance at the GDL/bipolar-plate interface (m²·K/W).
    """

    is_wet: bool = False

    def __post_init__(self):
        pass

    def set_catalyst_layer(self, cl):
        """Replace the catalyst layer."""
        self.cl = cl
        self.__post_init__()

    def set_gas_diffusion_layer(self, gdl):
        """Replace the gas diffusion layer."""
        self.gdl = gdl
        self.__post_init__()

    def set_channel(self, ch):
        """Replace the flow channel."""
        self.ch = ch
        self.__post_init__()


@dataclass
class FuelCell(Cell):
    """Proton exchange membrane fuel cell.

    The cell object is a pure component tree: it holds geometry and material
    parameters only.  All physics are evaluated by a separate model object::

        model = ExplicitSteadyStateModel()
        state = model.set_initial_conditions(cell, conditions)
        state = model.solve(cell, conditions, state)
        # state.cell_voltage, state.mea_temperature, … are now populated

    Attributes
    ----------
    ca : FuelCellSide
        Cathode side (catalyst layer, GDL, optional MPL, flow channel).
    an : FuelCellSide
        Anode side.
    membrane : Membrane
        Membrane component (e.g. :class:`~marapendi.membrane.pem.PFSA`).
    area : float
        Active cell area (m²).
    electrical_resistance : float
        Through-plane electrical contact resistance (Ω·m²).
    cell_number : int
        Number of cells in the stack (default 1 — single cell).
    mea_surface_heat_capacity : float
        Effective MEA heat capacity per unit area (J/m²/K).
    """

    ca: FuelCellSide = field(default_factory=FuelCellSide)
    an: FuelCellSide = field(default_factory=FuelCellSide)
    membrane: Membrane = field(default_factory=Membrane)

    cell_number: int = 1
    mea_surface_heat_capacity: float = 10000.
    use_eq_water_content_for_ionomer: bool = True

    def __post_init__(self):
        self.ca.reactant = 'o2'
        self.an.reactant = 'h2'
        super().__post_init__()
        self._voltage_model = VoltageModel()
        self._thermal_model = ThermalModel()
        self._model = ExplicitSteadyStateModel(
            self._voltage_model,
            self._thermal_model,
        )
        self._gas_transport_model = self._model.gas_transport_model
        self.state = CellState()

    # ------------------------------------------------------------------
    # Voltage accessors — require self.state to be populated by a solve call
    # ------------------------------------------------------------------

    def reversible_cell_voltage(self):
        return self._voltage_model.reversible_cell_voltage(self, self.state)

    def reversible_voltage_vs_RHE(self):
        return self._voltage_model.reversible_voltage_vs_RHE(self, self.state)

    def activation_overpotential(self, theta_PtO=0):
        return self._voltage_model.activation_overpotential(self, self.state, theta_PtO)

    def high_frequency_resistance(self):
        return self._voltage_model.high_frequency_resistance(self, self.state)

    def ohmic_overpotential(self):
        return self._voltage_model.ohmic_overpotential(self, self.state)

    def calculate_theta_PtO(self):
        return self._voltage_model.calculate_theta_PtO(self, self.state)

    def calculate_cell_voltage(self):
        return self._voltage_model.compute_cell_voltage(self, self.state)

    # ------------------------------------------------------------------
    # Transient solver support (internal — not part of steady-state API)
    # ------------------------------------------------------------------

    def set_water_saturation_in_porous_layers(self, saturation_profile):
        k = 0
        for side in (self.an, self.ca):
            for layer in side.porous_layers:
                layer.non_wetting_saturation = saturation_profile[k, ...]
                k += 1
        for side in (self.an, self.ca):
            side.h2ov_transport_resistance = self._gas_transport_model.gas_transport_resistance(
                side, side, 'h2o'
            )
            side.cl.set_water_film_thickness(side.cl.non_wetting_saturation)

    def f_transient(self, t, x, u, p, n_memb_mesh=3):
        """Right-hand side for the full transient ODE: [dT/dt | dλ/dt (membrane) | ds/dt (porous) | ds_relax/dt].

        The state vector *x* is ordered as
        ``[mea_temperature, *membrane_water_profile, *saturation_profile, s_relax_ca, s_relax_an]``.
        """
        self.set_conditions_from_input_dict(u, t * np.ones_like(x[0, ...]))
        k = 1 + n_memb_mesh
        water_profile = x[1:k, ...]
        saturation_profile = np.clip(x[k:k + len(self.porous_layers), ...], 0, 0.9)
        self.ca.s_relax = x[-2, ...]
        self.an.s_relax = x[-1, ...]

        self.set_mea_temperature(x[0, ...])
        self.set_water_saturation_in_porous_layers(saturation_profile)
        self._model.water_balance_model.solve_water_balance(self, water_profile=water_profile, dynamic=True)
        self.calculate_gas_concentrations_at_cl()
        self.calculate_cell_voltage()

        wbm = self._model.water_balance_model
        dlmbddt = wbm.membrane_water_rate_of_change(self, n_memb_mesh)
        dTdt = self._thermal_model.temperature_rate_of_change(self)
        dsdt = wbm.saturation_rate_of_change(self)
        dsrlxdt = wbm.relaxation_rate_of_change(self)
        return [dTdt] + list(dlmbddt) + list(dsdt) + list(dsrlxdt)

    def f_relax(self, t, x, u, p, n_memb_mesh=3):
        """Right-hand side for the ionomer relaxation sub-ODE only (``[ds_relax_ca/dt, ds_relax_an/dt]``)."""
        self.set_conditions_from_input_dict(u, t * np.ones_like(x[0, ...]))
        self._run_explicit_solve()

        self.ca.s_relax = x[-2, ...]
        self.an.s_relax = x[-1, ...]

        dsrlxdt = self._model.water_balance_model.relaxation_rate_of_change(self)
        return list(dsrlxdt)

    def _run_explicit_solve(self):
        """Run the explicit steady-state physics on ``self.state`` and sync attributes."""
        state = self.state
        state.thermal_resistance = self._thermal_model.heat_transfer_resistance(self)
        mea_temperature = self._thermal_model.mea_temperature(self, state)
        self._thermal_model.set_mea_temperature(mea_temperature, self, state)
        self._model.water_balance_model.calculate_water_transport(
            self, state, gas_transport_model=self._gas_transport_model
        )
        self._gas_transport_model.calculate_gas_concentrations(self, state)
        self._voltage_model.compute_cell_voltage(self, state)
        self._sync_state_to_fc()

    def _sync_state_to_fc(self):
        """Copy solved quantities from ``self.state`` to legacy ``FuelCell`` attributes."""
        s = self.state
        self.cell_voltage = s.cell_voltage
        self.mea_temperature = s.mea_temperature if s.mea_temperature is not None else self.temperature
        self.mea_temperature_increase = (
            s.mea_temperature_increase if s.mea_temperature_increase is not None else 0.
        )
        self.thermal_resistance = s.thermal_resistance if s.thermal_resistance is not None else self.thermal_resistance
        self.crossover_current = s.crossover_current if s.crossover_current is not None else 0.
        if s.membrane.h2_permeation_flux is not None:
            self.h2_permeation_flux = s.membrane.h2_permeation_flux

    def set_conditions_from_input_functions(self, u, t):
        """Evaluate time-dependent input functions *u* at time *t* and update the cell state."""
        self.set_conditions_from_input_dict({key: fn(t) for key, fn in u.items()})

    def set_conditions_from_input_dict(self, u):
        """Build a :class:`CellState` from the flat input dict *u* (keys are e.g. ``'current-density'``)."""
        current_density = u['current-density']
        stack_temperature = u['cell-temperature']
        cathode_conditions, anode_conditions = [
            SideConditions(
                inlet_temperature=u[f'{side}-inlet-temperature'],
                inlet_pressure=u[f'{side}-inlet-pressure'],
                outlet_pressure=u[f'{side}-outlet-pressure'],
                inlet_relative_humidity=u[f'{side}-inlet-rh'],
                stoichiometry=u[f'{side}-stoichiometry'],
                dry_o2_mole_fraction=u[f'{side}-dry-o2-mole-fraction'],
                dry_h2_mole_fraction=u[f'{side}-dry-h2-mole-fraction'],
                inlet_liquid_saturation=u[f'{side}-inlet-liquid-saturation'],
                inlet_gas_flow_rate=u[f'{side}-inlet-gas-flow-rate'],
                inlet_liquid_flow_rate=u[f'{side}-inlet-liquid-flow-rate'],
                inlet_liquid=u.get(f'{side}-inlet-liquid', ElectrolyteSolution()),
            )
            for side in ('ca', 'an')
        ]
        self.state = self._model._init_state(
            self, stack_temperature, current_density,
            cathode_conditions, anode_conditions,
        )


from ..simulation.conditions import SideConditions, OperatingConditions, DynamicOperatingConditions  # noqa: F401
