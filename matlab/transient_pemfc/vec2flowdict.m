function d = vec2flowdict(vec)
%VEC2FLOWDICT Convert a flattened GasFlowState vector (order =
%   gasflow_field_order()) into the py.dict shape expected by
%   marapendi.interop.simulink_bridge (a ca_flow/an_flow argument).

    names = gasflow_field_order();
    vec = double(vec(:));
    args = cell(1, 2 * numel(names));
    args(1:2:end) = names;
    args(2:2:end) = num2cell(vec);
    d = py.dict(pyargs(args{:}));
end
