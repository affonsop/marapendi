function create_buses(n_memb_mesh)
%CREATE_BUSES Define the Simulink.Bus objects used by the TransientPEMFC block.
%
%   create_buses(n_memb_mesh) creates (in the base workspace):
%     - GasFlowStateBus : one GasFlowState (mirrors
%                         marapendi.simulation.state.GasFlowState), used for
%                         both the ca/an inlet inputs and the ca/an outlet outputs
%     - CellStateBus    : flattened diagnostics (mirrors the dict returned
%                         by marapendi.interop.simulink_bridge.cell_diagnostics,
%                         minus the ca_outlet_*/an_outlet_* fields, which are
%                         carried on their own GasFlowStateBus outputs instead)
%
%   n_memb_mesh sizes the membrane_water_content_profile field and must match
%   the value baked into the S-Function mask.

    if nargin < 1
        n_memb_mesh = 5;
    end

    % ---- GasFlowStateBus ---------------------------------------------------
    flowElems = local_scalar_elements(gasflow_field_order());
    flowBus = Simulink.Bus;
    flowBus.Elements = flowElems;
    assignin('base', 'GasFlowStateBus', flowBus);

    % ---- CellStateBus --------------------------------------------------
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

    fprintf('create_buses: GasFlowStateBus, CellStateBus (n_memb_mesh=%d) in base workspace\n', n_memb_mesh);
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
