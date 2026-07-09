function d = vec2conddict(vec)
%VEC2CONDDICT Convert a flattened CellConditions vector (order =
%   cond_field_order()) into the py.dict shape expected by
%   marapendi.interop.simulink_bridge.

    names = cond_field_order();
    vec = double(vec(:));
    args = cell(1, 2 * numel(names));
    args(1:2:end) = names;
    args(2:2:end) = num2cell(vec);
    d = py.dict(pyargs(args{:}));
end
