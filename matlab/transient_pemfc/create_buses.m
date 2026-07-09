function create_buses(n_memb_mesh)
%CREATE_BUSES Define the Simulink.Bus objects used by the TransientPEMFC block.
%
%   create_buses(n_memb_mesh) creates (in the base workspace):
%     - SideConditionsBus   : one side's inlet boundary conditions (mirrors
%                              marapendi.simulation.conditions.SideConditions)
%     - CellConditionsBus   : current_density, cell_temperature, ca, an
%                              (mirrors marapendi.simulation.conditions.CellConditions)
%     - CellStateBus        : flattened diagnostics (mirrors the dict returned
%                              by marapendi.interop.simulink_bridge.diagnostics)
%
%   n_memb_mesh sizes the membrane_water_content_profile field and must match
%   the value baked into the S-Function mask.

    if nargin < 1
        n_memb_mesh = 5;
    end

    % ---- SideConditionsBus ------------------------------------------------
    condFields = cond_field_order();
    sideFields = strrep(condFields(startsWith(condFields, 'ca_')), 'ca_', '');
    sideElems = local_scalar_elements(sideFields);
    sideBus = Simulink.Bus;
    sideBus.Elements = sideElems;
    assignin('base', 'SideConditionsBus', sideBus);

    % ---- CellConditionsBus -------------------------------------------------
    condElems = [
        local_scalar_elements({'current_density'; 'cell_temperature'})
        local_bus_element('ca', 'SideConditionsBus')
        local_bus_element('an', 'SideConditionsBus')
    ];
    condBus = Simulink.Bus;
    condBus.Elements = condElems;
    assignin('base', 'CellConditionsBus', condBus);

    % ---- CellStateBus --------------------------------------------------
    % Scalar diagnostic fields, matching simulink_bridge.diagnostics() keys
    % (all but membrane_water_content_profile, which is a vector).
    stateElems = local_scalar_elements(state_scalar_field_order());

    profileElem = Simulink.BusElement;
    profileElem.Name = 'membrane_water_content_profile';
    profileElem.Dimensions = n_memb_mesh;
    profileElem.DataType = 'double';
    profileElem.Complexity = 'real';
    profileElem.SampleTime = -1;

    stateBus = Simulink.Bus;
    stateBus.Elements = [stateElems; profileElem];
    assignin('base', 'CellStateBus', stateBus);

    fprintf('create_buses: SideConditionsBus, CellConditionsBus, CellStateBus (n_memb_mesh=%d) in base workspace\n', n_memb_mesh);
end

function elems = local_scalar_elements(names)
    elems = Simulink.BusElement.empty(0, 1);
    for k = 1:numel(names)
        e = Simulink.BusElement;
        e.Name = names{k};
        e.Dimensions = 1;
        e.DataType = 'double';
        e.Complexity = 'real';
        e.SampleTime = -1;
        elems(k, 1) = e; %#ok<AGROW>
    end
end

function e = local_bus_element(name, busName)
    e = Simulink.BusElement;
    e.Name = name;
    e.Dimensions = 1;
    e.DataType = ['Bus: ' busName];
    e.Complexity = 'real';
    e.SampleTime = -1;
end
