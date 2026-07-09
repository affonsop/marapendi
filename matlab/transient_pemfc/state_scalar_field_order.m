function names = state_scalar_field_order()
%STATE_SCALAR_FIELD_ORDER Fixed 26-entry order of the scalar CellState
%   diagnostic fields (everything in CellStateBus except
%   membrane_water_content_profile, which is a vector appended separately).
%   Must match marapendi.interop.simulink_bridge.diagnostics()'s dict keys.

    names = {
        'cell_voltage'
        'mea_temperature'
        'thermal_resistance'
        'hfr'
        'E_rev'
        'eta_act'
        'eta_ohm'
        'crossover_current'
        'membrane_water_content'
        'membrane_water_flux'
        'membrane_h2_permeation_flux'
        'membrane_proton_resistance'
        'ca_cl_ionomer_water_content'
        'ca_cl_liquid_saturation'
        'ca_cl_proton_resistance'
        'ca_water_flux'
        'ca_liquid_flux'
        'ca_membrane_water_flux'
        'ca_h2ov_transport_resistance'
        'an_cl_ionomer_water_content'
        'an_cl_liquid_saturation'
        'an_cl_proton_resistance'
        'an_water_flux'
        'an_liquid_flux'
        'an_membrane_water_flux'
        'an_h2ov_transport_resistance'
    };
end
