"""Flat scalar/list interface to :class:`~marapendi.models.base.transient.TransientModel`,
called from MATLAB/Simulink via ``py.marapendi.interop.simulink_bridge.*``.

MATLAB's Python interface marshals plain scalars, lists, and dicts of those
unambiguously; it does not need to know anything about marapendi's dataclasses.
Every function here therefore takes/returns only ``float``, ``list[float]``, and
``dict`` — the corresponding Simulink S-Function (``matlab/transient_pemfc/
transient_pemfc_sfun.m``) builds/reads a ``py.dict`` of operating-condition and
diagnostic fields whose keys line up 1:1 with the dict keys used below.

Editing the physics? Edit :mod:`marapendi.models.base.transient` and friends as
usual — this module is a thin adapter, not a copy of the model.
"""
from __future__ import annotations

import numpy as np

from ..cell.fuelcell import FuelCell
from ..channel.flow_channels import FlowChannel
from ..channel.gas_transport_resistance import ChannelGasResistanceModel
from ..porous_layers.porous_layers import GasDiffusionLayer, MicroPorousLayer
from ..porous_layers.catalyst_layers import PtCCatalystLayer
from ..models.darcy import DarcyTransportModel
from ..models.thermo.electrochemistry import ElectrochemicalReaction
from ..membrane.pem import PFSAIonomer, PFSA
from ..simulation.conditions import CellConditions, SideConditions
from ..models.base.transient import TransientModel

_SIDE_FIELDS = (
    'inlet_temperature', 'inlet_pressure', 'outlet_pressure',
    'dry_o2_mole_fraction', 'dry_h2_mole_fraction', 'inlet_relative_humidity',
    'stoichiometry', 'inlet_liquid_saturation', 'inlet_liquid_flow_rate',
    'inlet_gas_flow_rate', 'minimum_current_density_for_stoich',
)

_model_cache: dict[int, TransientModel] = {}


def build_default_cell() -> FuelCell:
    """Assemble the reference cell used in ``examples/plot_01_polarization_curve.py``.

    Returns a fresh :class:`~marapendi.cell.fuelcell.FuelCell` instance; callers
    that want a different cell should either edit this function or build their
    own ``FuelCell`` in Python and hand it to the MATLAB block's mask.
    """
    cell = FuelCell(area=25e-4, electric_resistance=10e-7)

    liq = DarcyTransportModel(J_function_exponent=0.4)

    for side in cell.sides:
        side.ch = FlowChannel(
            width=0.85e-3, height=1e-3, length=0.49, n_parallel=3,
            reactant="o2" if side is cell.ca else "h2",
            transport_resistance_model=ChannelGasResistanceModel(sherwood=3.66, B_ch=1.2),
        )
        side.thermal_contact_resistance = 1e-4
        side.gdl = GasDiffusionLayer(
            thickness=117e-6 * 1.4, porosity=0.65, tortuosity=1.55,
            contact_angle=110.0, absolute_permeability=3e-12,
            thermal_conductivity=1.2, two_phase_transport_model=liq,
            relative_permeability_exponent=3, volume_heat_capacity=1.58e6,
        )
        side.mpl = MicroPorousLayer(
            thickness=22e-6, porosity=0.4, tortuosity=3, pore_diameter=500e-9,
            contact_angle=130.0, absolute_permeability=1e-12,
            thermal_conductivity=0.144, two_phase_transport_model=liq,
            relative_permeability_exponent=3, volume_heat_capacity=1.98e6,
        )

    orr = ElectrochemicalReaction(
        reference_exchange_current_density=1e-3, reaction_order=0.8,
        activation_energy=42e6, reference_activity=1e5,
        reference_temperature=353.15, number_of_electrons=2,
        charge_transfer_coeff=0.5,
    )

    nafion = PFSAIonomer(
        equivalent_weight=1100., dry_density=1980,
        reference_conductivity=50., residual_conductivity=0.3,
        conductivity_fv_threshold=0.04, conductivity_exp=1.5,
        reference_conductivity_temperature=300.,
        conductivity_activation_energy=10.540e6,
        reference_water_absorption_coefficient=1e-5,
        reference_water_absorption_temperature=303.15,
        water_absorption_activation_energy=20e6,
        reference_water_diffusivity=2e-10,
        reference_water_diffusivity_temperature=300.,
        water_diffusivity_activation_energy=20e6,
        vapor_equilibrium_polynomial=[36, -39.85, 17.18, 0.043],
    )

    cell.ca.cl = PtCCatalystLayer(
        ecsa=40e3, platinum_loading=0.5e-2, catalyst_platinum_weight_percent=0.5,
        ionomer_to_carbon_ratio=0.81, ionomer=nafion, reaction=orr,
        thickness=10e-6, tortuosity=3, thermal_conductivity=0.18,
        pore_diameter=140e-9, carbon_agglomerate_radius=25e-9,
        absolute_permeability=2e-13, contact_angle=100.0,
        two_phase_transport_model=liq, relative_permeability_exponent=3,
        volume_heat_capacity=1.56e6,
    )
    cell.an.cl = PtCCatalystLayer(
        platinum_loading=0.1e-2, ionomer=nafion, thickness=7e-6,
        ionomer_to_carbon_ratio=0.57, catalyst_platinum_weight_percent=0.2,
        thermal_conductivity=0.18, pore_diameter=140e-9,
        absolute_permeability=1e-13, contact_angle=100.0,
        two_phase_transport_model=liq, volume_heat_capacity=1.56e6,
    )

    cell.membrane = PFSA(ionomer=nafion, dry_thickness=15e-6)

    return cell


def _get_model(n_memb_mesh: int) -> TransientModel:
    n_memb_mesh = int(n_memb_mesh)
    if n_memb_mesh not in _model_cache:
        _model_cache[n_memb_mesh] = TransientModel(n_memb_mesh=n_memb_mesh)
    return _model_cache[n_memb_mesh]


def _side_conditions(prefix: str, cond: dict) -> SideConditions:
    return SideConditions(**{
        field: float(cond[f'{prefix}_{field}']) for field in _SIDE_FIELDS
    })


def _cell_conditions(cond: dict) -> CellConditions:
    return CellConditions(
        current_density=float(cond['current_density']),
        cell_temperature=float(cond['cell_temperature']),
        ca=_side_conditions('ca', cond),
        an=_side_conditions('an', cond),
    )


def initial_state(cell: FuelCell, n_memb_mesh: int, cond: dict) -> list:
    """Steady-state ODE initial condition, as a plain list (mirrors ``x0``)."""
    model = _get_model(n_memb_mesh)
    _, x0 = model.set_initial_conditions(cell, _cell_conditions(cond))
    return x0.tolist()


def derivative(cell: FuelCell, n_memb_mesh: int, t: float, x: list, cond: dict) -> list:
    """ODE right-hand side at time *t*, state *x* — mirrors ``f_transient``."""
    model = _get_model(n_memb_mesh)
    dxdt = model.f_transient(float(t), np.asarray(x, dtype=float), cell, _cell_conditions(cond))
    return np.asarray(dxdt, dtype=float).tolist()


def _s(v):
    return float(np.atleast_1d(v)[0]) if v is not None else float('nan')


def _state_to_dict(state) -> dict:
    """Flatten a scalar :class:`~marapendi.simulation.state.CellState` — the
    same field list ``diagnostics()`` and ``step()`` both return, so the
    Simulink side (``state_scalar_field_order.m``) only has one shape to match."""
    return {
        'cell_voltage': _s(state.cell_voltage),
        'mea_temperature': _s(state.mea_temperature),
        'thermal_resistance': _s(state.thermal_resistance),
        'hfr': _s(state.hfr),
        'E_rev': _s(state.E_rev),
        'eta_act': _s(state.eta_act),
        'eta_ohm': _s(state.eta_ohm),
        'crossover_current': _s(state.crossover_current),
        'membrane_water_content': _s(state.membrane.water_content),
        'membrane_water_content_profile': np.asarray(state.membrane.water_content_profile, dtype=float).reshape(-1).tolist(),
        'membrane_water_flux': _s(state.membrane.water_flux),
        'membrane_h2_permeation_flux': _s(state.membrane.h2_permeation_flux),
        'membrane_proton_resistance': _s(state.membrane.proton_resistance),
        'ca_cl_ionomer_water_content': _s(state.ca.cl.ionomer_water_content),
        'ca_cl_liquid_saturation': _s(state.ca.cl.liquid_saturation),
        'ca_cl_proton_resistance': _s(state.ca.cl.proton_resistance),
        'ca_water_flux': _s(state.ca.water_flux),
        'ca_liquid_flux': _s(state.ca.liquid_flux),
        'ca_membrane_water_flux': _s(state.ca.membrane_water_flux),
        'ca_h2ov_transport_resistance': _s(state.ca.h2ov_transport_resistance),
        'an_cl_ionomer_water_content': _s(state.an.cl.ionomer_water_content),
        'an_cl_liquid_saturation': _s(state.an.cl.liquid_saturation),
        'an_cl_proton_resistance': _s(state.an.cl.proton_resistance),
        'an_water_flux': _s(state.an.water_flux),
        'an_liquid_flux': _s(state.an.liquid_flux),
        'an_membrane_water_flux': _s(state.an.membrane_water_flux),
        'an_h2ov_transport_resistance': _s(state.an.h2ov_transport_resistance),
    }


def diagnostics(cell: FuelCell, n_memb_mesh: int, t: float, x: list, cond: dict) -> dict:
    """Flattened :class:`~marapendi.simulation.state.CellState` at (*t*, *x*) — mirrors ``evaluate``."""
    model = _get_model(n_memb_mesh)
    x_arr = np.asarray(x, dtype=float).reshape(-1, 1)
    state = model.evaluate(cell, _cell_conditions(cond), np.array([float(t)]), x_arr)
    return _state_to_dict(state)


def step(cell: FuelCell, n_memb_mesh: int, t: float, x: list, cond: dict) -> dict:
    """``derivative()`` and ``diagnostics()`` combined into a single Python
    round trip and a single physics pass, via
    :meth:`~marapendi.models.base.transient.TransientModel.f_transient`'s
    ``return_state=True``. Returns ``{'dxdt': [...], **diagnostics_fields}``.
    Intended for callers (e.g. a Simulink S-Function) that need both the ODE
    right-hand side and diagnostics at the same point and want to avoid
    computing the physics twice — ``TransientModel.solve()`` itself still
    calls ``evaluate()`` separately after integration, since
    :func:`scipy.integrate.solve_ivp` has no hook to carry state out of the
    right-hand-side function.
    """
    model = _get_model(n_memb_mesh)
    dxdt, state = model.f_transient(
        float(t), np.asarray(x, dtype=float), cell, _cell_conditions(cond), return_state=True)
    out = _state_to_dict(state)
    out['dxdt'] = np.asarray(dxdt, dtype=float).tolist()
    return out
