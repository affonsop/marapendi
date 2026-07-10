function names = gasflow_field_order()
%GASFLOW_FIELD_ORDER Fixed 7-entry field order for a flattened GasFlowState
%   vector (temperature, pressure, then species in O2/N2/H2/H2O order,
%   then liquid). Must match marapendi.interop.simulink_bridge.GASFLOW_FIELDS.

    names = {'temperature'; 'pressure'; 'o2'; 'n2'; 'h2'; 'h2o'; 'liquid'};
end
