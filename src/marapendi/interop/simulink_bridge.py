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

from ..components.cell.fuelcell import FuelCell
from ..components.channel.flow_channels import FlowChannel
from ..models.channel import ChannelGasResistanceModel
from ..components.porous_layers.porous_layers import GasDiffusionLayer, MicroPorousLayer
from ..components.porous_layers.catalyst_layers import PtCCatalystLayer
from ..models.darcy import DarcyTransportModel
from ..models.thermo.electrochemistry import ElectrochemicalReaction
from ..models.thermo.gas import index_o2, index_n2, index_h2, index_h2ov
from ..components.membrane.pem import PFSAIonomer, PFSA
from ..simulation.conditions import CellConditions
from ..simulation.state import GasFlowState
from ..models.base.transient import TransientModel

_model_cache: dict[int, TransientModel] = {}


def default_cell_params() -> dict:
    """Nested dict of every parameter :func:`build_cell_from_params` accepts,
    with the same default values as the reference cell in
    ``examples/plot_01_polarization_curve.py``.

    From MATLAB, ``matlab/transient_pemfc/cell_params_template.m`` is the
    struct-shaped mirror of this dict (copy it, edit the values, and point
    the ``TransientPEMFC`` block's ``cellBuilderExpr`` mask parameter at your
    copy — see ``call_python_builder.m``). Both are kept in sync by hand;
    if you add a parameter to one, add it to the other.
    """
    return {
        'area': 25e-4,
        'electric_resistance': 10e-7,
        'thermal_contact_resistance': 1e-4,
        'channel': {
            'width': 0.85e-3, 'height': 1e-3, 'length': 0.49, 'n_parallel': 3,
            'sherwood': 3.66, 'B_ch': 1.2,
        },
        'gdl': {
            'thickness': 117e-6 * 1.4, 'porosity': 0.65, 'tortuosity': 1.55,
            'contact_angle': 110.0, 'absolute_permeability': 3e-12,
            'thermal_conductivity': 1.2, 'relative_permeability_exponent': 3,
            'volume_heat_capacity': 1.58e6,
            'J_function_exponent': 0.4,  # shared by gdl/mpl/cl two-phase transport
        },
        'mpl': {
            'thickness': 22e-6, 'porosity': 0.4, 'tortuosity': 3, 'pore_diameter': 500e-9,
            'contact_angle': 130.0, 'absolute_permeability': 1e-12,
            'thermal_conductivity': 0.144, 'relative_permeability_exponent': 3,
            'volume_heat_capacity': 1.98e6,
        },
        'orr': {
            'reference_exchange_current_density': 1e-3, 'reaction_order': 0.8,
            'activation_energy': 42e6, 'reference_activity': 1e5,
            'reference_temperature': 353.15, 'number_of_electrons': 2,
            'charge_transfer_coeff': 0.5,
        },
        'ionomer': {
            'equivalent_weight': 1100., 'dry_density': 1980.,
            'reference_conductivity': 50., 'residual_conductivity': 0.3,
            'conductivity_fv_threshold': 0.04, 'conductivity_exp': 1.5,
            'reference_conductivity_temperature': 300.,
            'conductivity_activation_energy': 10.540e6,
            'reference_water_absorption_coefficient': 1e-5,
            'reference_water_absorption_temperature': 303.15,
            'water_absorption_activation_energy': 20e6,
            'reference_water_diffusivity': 2e-10,
            'reference_water_diffusivity_temperature': 300.,
            'water_diffusivity_activation_energy': 20e6,
            'vapor_equilibrium_polynomial': [36, -39.85, 17.18, 0.043],
        },
        'ca_cl': {
            'ecsa': 40e3, 'platinum_loading': 0.5e-2, 'catalyst_platinum_weight_percent': 0.5,
            'ionomer_to_carbon_ratio': 0.81, 'thickness': 10e-6, 'tortuosity': 3,
            'thermal_conductivity': 0.18, 'pore_diameter': 140e-9,
            'carbon_agglomerate_radius': 25e-9, 'absolute_permeability': 2e-13,
            'contact_angle': 100.0, 'relative_permeability_exponent': 3,
            'volume_heat_capacity': 1.56e6,
        },
        'an_cl': {
            'platinum_loading': 0.1e-2, 'thickness': 7e-6,
            'ionomer_to_carbon_ratio': 0.57, 'catalyst_platinum_weight_percent': 0.2,
            'thermal_conductivity': 0.18, 'pore_diameter': 140e-9,
            'absolute_permeability': 1e-13, 'contact_angle': 100.0,
            'volume_heat_capacity': 1.56e6,
        },
        'membrane': {'dry_thickness': 15e-6},
    }


def _merge_cell_params(overrides: dict | None) -> dict:
    """Shallow-merge *overrides* onto :func:`default_cell_params`, one level
    into each top-level group (e.g. ``{'ca_cl': {'thickness': 12e-6}}``
    overrides just that field, keeping the rest of ``ca_cl``'s defaults)."""
    merged = {k: (dict(v) if isinstance(v, dict) else v) for k, v in default_cell_params().items()}
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def build_cell_from_params(params: dict | None = None) -> FuelCell:
    """Assemble a :class:`~marapendi.components.cell.fuelcell.FuelCell` from a (possibly
    partial) nested dict of parameters, filling in anything not given from
    :func:`default_cell_params`.

    This is what :func:`build_default_cell` calls with no overrides. Called
    from MATLAB with a user-edited struct (see ``cell_params_template.m``)
    converted to a nested ``py.dict`` by ``matstruct2pydict.m``.
    """
    p = _merge_cell_params(params)

    cell = FuelCell(area=p['area'], electric_resistance=p['electric_resistance'])
    liq = DarcyTransportModel(J_function_exponent=p['gdl']['J_function_exponent'])

    for side in cell.sides:
        side.ch = FlowChannel(
            width=p['channel']['width'], height=p['channel']['height'],
            length=p['channel']['length'], n_parallel=p['channel']['n_parallel'],
            reactant="o2" if side is cell.ca else "h2",
            transport_resistance_model=ChannelGasResistanceModel(
                sherwood=p['channel']['sherwood'], B_ch=p['channel']['B_ch']),
        )
        side.thermal_contact_resistance = p['thermal_contact_resistance']
        side.gdl = GasDiffusionLayer(
            thickness=p['gdl']['thickness'], porosity=p['gdl']['porosity'],
            tortuosity=p['gdl']['tortuosity'], contact_angle=p['gdl']['contact_angle'],
            absolute_permeability=p['gdl']['absolute_permeability'],
            thermal_conductivity=p['gdl']['thermal_conductivity'],
            two_phase_transport_model=liq,
            relative_permeability_exponent=p['gdl']['relative_permeability_exponent'],
            volume_heat_capacity=p['gdl']['volume_heat_capacity'],
        )
        side.mpl = MicroPorousLayer(
            thickness=p['mpl']['thickness'], porosity=p['mpl']['porosity'],
            tortuosity=p['mpl']['tortuosity'], pore_diameter=p['mpl']['pore_diameter'],
            contact_angle=p['mpl']['contact_angle'],
            absolute_permeability=p['mpl']['absolute_permeability'],
            thermal_conductivity=p['mpl']['thermal_conductivity'],
            two_phase_transport_model=liq,
            relative_permeability_exponent=p['mpl']['relative_permeability_exponent'],
            volume_heat_capacity=p['mpl']['volume_heat_capacity'],
        )

    orr = ElectrochemicalReaction(**p['orr'])
    ionomer = PFSAIonomer(**p['ionomer'])

    cell.ca.cl = PtCCatalystLayer(
        ecsa=p['ca_cl']['ecsa'], platinum_loading=p['ca_cl']['platinum_loading'],
        catalyst_platinum_weight_percent=p['ca_cl']['catalyst_platinum_weight_percent'],
        ionomer_to_carbon_ratio=p['ca_cl']['ionomer_to_carbon_ratio'],
        ionomer=ionomer, reaction=orr, thickness=p['ca_cl']['thickness'],
        tortuosity=p['ca_cl']['tortuosity'], thermal_conductivity=p['ca_cl']['thermal_conductivity'],
        pore_diameter=p['ca_cl']['pore_diameter'],
        carbon_agglomerate_radius=p['ca_cl']['carbon_agglomerate_radius'],
        absolute_permeability=p['ca_cl']['absolute_permeability'], contact_angle=p['ca_cl']['contact_angle'],
        two_phase_transport_model=liq, relative_permeability_exponent=p['ca_cl']['relative_permeability_exponent'],
        volume_heat_capacity=p['ca_cl']['volume_heat_capacity'],
    )
    cell.an.cl = PtCCatalystLayer(
        platinum_loading=p['an_cl']['platinum_loading'], ionomer=ionomer,
        thickness=p['an_cl']['thickness'], ionomer_to_carbon_ratio=p['an_cl']['ionomer_to_carbon_ratio'],
        catalyst_platinum_weight_percent=p['an_cl']['catalyst_platinum_weight_percent'],
        thermal_conductivity=p['an_cl']['thermal_conductivity'], pore_diameter=p['an_cl']['pore_diameter'],
        absolute_permeability=p['an_cl']['absolute_permeability'], contact_angle=p['an_cl']['contact_angle'],
        two_phase_transport_model=liq, volume_heat_capacity=p['an_cl']['volume_heat_capacity'],
    )

    cell.membrane = PFSA(ionomer=ionomer, dry_thickness=p['membrane']['dry_thickness'])

    return cell


def build_default_cell() -> FuelCell:
    """Assemble the reference cell used in ``examples/plot_01_polarization_curve.py``.

    Returns a fresh :class:`~marapendi.components.cell.fuelcell.FuelCell` instance built
    from :func:`default_cell_params`'s defaults, i.e. ``build_cell_from_params()``
    with no overrides. Callers that want a different cell should use
    :func:`build_cell_from_params` (from MATLAB: copy and edit
    ``cell_params_template.m``) rather than editing this function in place.
    """
    return build_cell_from_params()


def _get_model(n_memb_mesh: int) -> TransientModel:
    n_memb_mesh = int(n_memb_mesh)
    if n_memb_mesh not in _model_cache:
        _model_cache[n_memb_mesh] = TransientModel(n_memb_mesh=n_memb_mesh)
    return _model_cache[n_memb_mesh]


def _s(v):
    return float(np.atleast_1d(v)[0]) if v is not None else float('nan')


def _state_to_dict(state) -> dict:
    """Flatten a scalar :class:`~marapendi.simulation.state.CellState`, used
    by ``cell_diagnostics()`` — so the Simulink side
    (``state_scalar_field_order.m``) only has one shape to match."""
    return {
        'cell_voltage': _s(state.cell_voltage),
        'mea_temperature': _s(state.mea_temperature),
        'thermal_resistance': _s(state.thermal_resistance),
        'hfr': _s(state.hfr),
        'E_rev': _s(state.E_rev),
        'eta_act': _s(state.eta_act),
        'eta_ohm': _s(state.eta_ohm),
        'crossover_current': _s(state.crossover_current),
        'heat_release': _s(state.heat_release),
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


GASFLOW_FIELDS = ('temperature', 'pressure', 'o2', 'n2', 'h2', 'h2o', 'liquid')
"""Flat field order for a :class:`~marapendi.simulation.state.GasFlowState`
dict, mirrored by ``matlab/transient_pemfc/gasflow_field_order.m``."""


def _gas_flow_state_from_dict(flow: dict) -> GasFlowState:
    """Build a :class:`~marapendi.simulation.state.GasFlowState` from a flat
    dict with keys :data:`GASFLOW_FIELDS` (species order O2, N2, H2, H2O,
    matching :data:`~marapendi.models.thermo.gas.species_indexes`)."""
    return GasFlowState(
        temperature=float(flow['temperature']),
        pressure=float(flow['pressure']),
        gas_species_molar_flow_rates=np.array([
            float(flow['o2']), float(flow['n2']), float(flow['h2']), float(flow['h2o']),
        ]),
        liquid_molar_flow_rate=float(flow['liquid']),
    )


def _gas_flow_state_to_dict(flow_state: GasFlowState, prefix: str) -> dict:
    """Flatten a :class:`~marapendi.simulation.state.GasFlowState` into a
    dict with keys ``{prefix}_<GASFLOW_FIELDS>``."""
    X = flow_state.gas_species_molar_flow_rates
    return {
        f'{prefix}_temperature': _s(flow_state.temperature),
        f'{prefix}_pressure': _s(flow_state.pressure),
        f'{prefix}_o2': _s(X[index_o2]),
        f'{prefix}_n2': _s(X[index_n2]),
        f'{prefix}_h2': _s(X[index_h2]),
        f'{prefix}_h2o': _s(X[index_h2ov]),
        f'{prefix}_liquid': _s(flow_state.liquid_molar_flow_rate),
    }


def _cell_conditions_from_flows(ca_flow: dict, an_flow: dict,
                                 current_density: float, cell_temperature: float) -> CellConditions:
    return CellConditions(
        current_density=float(current_density),
        cell_temperature=float(cell_temperature),
        ca=_gas_flow_state_from_dict(ca_flow).to_side_conditions(),
        an=_gas_flow_state_from_dict(an_flow).to_side_conditions(),
    )


def cell_initial_state(cell: FuelCell, n_memb_mesh: int, ca_flow: dict, an_flow: dict,
                        current_density: float, cell_temperature: float) -> list:
    """Steady-state ODE initial condition from inlet ``GasFlowState`` dicts
    + current density + cell temperature, as a plain list (mirrors ``x0``)."""
    model = _get_model(n_memb_mesh)
    cond = _cell_conditions_from_flows(ca_flow, an_flow, current_density, cell_temperature)
    _, x0 = model.set_initial_conditions(cell, cond)
    return x0.tolist()


def cell_derivative(cell: FuelCell, n_memb_mesh: int, t: float, x: list, ca_flow: dict, an_flow: dict,
                     current_density: float, cell_temperature: float) -> list:
    """ODE right-hand side at time *t*, state *x*, driven by inlet
    ``GasFlowState`` dicts + current density + cell temperature — mirrors
    :meth:`~marapendi.models.base.transient.TransientModel.f_transient`."""
    model = _get_model(n_memb_mesh)
    cond = _cell_conditions_from_flows(ca_flow, an_flow, current_density, cell_temperature)
    dxdt = model.f_transient(float(t), np.asarray(x, dtype=float), cell, cond)
    return np.asarray(dxdt, dtype=float).tolist()


def cell_diagnostics(cell: FuelCell, n_memb_mesh: int, t: float, x: list, ca_flow: dict, an_flow: dict,
                      current_density: float, cell_temperature: float) -> dict:
    """Flattened :class:`~marapendi.simulation.state.CellState` at (*t*, *x*),
    driven by inlet ``GasFlowState`` dicts + current density + cell
    temperature — mirrors :meth:`~marapendi.models.base.transient.TransientModel.evaluate`,
    plus the outlet ``GasFlowState``s (``ca_outlet_*``/``an_outlet_*`` keys,
    see :data:`GASFLOW_FIELDS`) computed by
    :meth:`~marapendi.models.base.explicit_steady_state.ExplicitSteadyStateModel.set_gas_flow_states`'s
    mass balance."""
    model = _get_model(n_memb_mesh)
    cond = _cell_conditions_from_flows(ca_flow, an_flow, current_density, cell_temperature)
    x_arr = np.asarray(x, dtype=float).reshape(-1, 1)
    state = model.evaluate(cell, cond, np.array([float(t)]), x_arr)
    out = _state_to_dict(state)
    out.update(_gas_flow_state_to_dict(state.ca.outlet_gas_flow_state, 'ca_outlet'))
    out.update(_gas_flow_state_to_dict(state.an.outlet_gas_flow_state, 'an_outlet'))
    return out
