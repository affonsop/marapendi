function v = diagdict2flowvec(diagDict, prefix)
%DIAGDICT2FLOWVEC Extract a 7-element GasFlowState vector (order =
%   gasflow_field_order()) for keys named ``{prefix}_<field>`` from the
%   py.dict returned by marapendi.interop.simulink_bridge.cell_diagnostics()
%   (prefix is 'ca_outlet' or 'an_outlet').

    names = gasflow_field_order();
    v = zeros(numel(names), 1);
    for k = 1:numel(names)
        v(k) = double(diagDict{[prefix '_' names{k}]});
    end
end
