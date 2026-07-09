function names = cond_field_order()
%COND_FIELD_ORDER Fixed 24-entry field order for the flattened CellConditions
%   vector used between the Bus Selector and the TransientPEMFC S-Function.
%   Must match marapendi.interop.simulink_bridge's expected dict keys.

    sideFields = {
        'inlet_temperature'
        'inlet_pressure'
        'outlet_pressure'
        'dry_o2_mole_fraction'
        'dry_h2_mole_fraction'
        'inlet_relative_humidity'
        'stoichiometry'
        'inlet_liquid_saturation'
        'inlet_liquid_flow_rate'
        'inlet_gas_flow_rate'
        'minimum_current_density_for_stoich'
    };
    names = [{'current_density'; 'cell_temperature'}; ...
             strcat('ca_', sideFields); ...
             strcat('an_', sideFields)];
end
