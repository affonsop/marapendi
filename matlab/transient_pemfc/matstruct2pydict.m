function d = matstruct2pydict(s)
%MATSTRUCT2PYDICT Recursively convert a scalar MATLAB struct into a py.dict.
%
%   Nested scalar structs become nested py.dict (matching the grouping in
%   marapendi.interop.simulink_bridge.default_cell_params(): 'channel',
%   'gdl', 'mpl', 'orr', 'ionomer', 'ca_cl', 'an_cl', 'membrane'). Numeric
%   vectors with more than one element become py.list (e.g. the ionomer's
%   'vapor_equilibrium_polynomial'); numeric scalars stay plain doubles.
%
%   Used by call_python_builder.m to pass a user-edited cell_params_template.m
%   struct to marapendi.interop.simulink_bridge.build_cell_from_params().

    fields = fieldnames(s);
    args = {};
    for i = 1:numel(fields)
        key = fields{i};
        val = s.(key);
        if isstruct(val)
            val = matstruct2pydict(val);
        elseif isnumeric(val) && numel(val) > 1
            val = py.list(val(:)');
        elseif isnumeric(val)
            val = double(val);
        end
        args = [args, {key, val}]; %#ok<AGROW>
    end
    d = py.dict(pyargs(args{:}));
end
