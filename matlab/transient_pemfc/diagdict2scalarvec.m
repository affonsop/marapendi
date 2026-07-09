function v = diagdict2scalarvec(diagDict)
%DIAGDICT2SCALARVEC Extract the 26 scalar diagnostic fields (order =
%   state_scalar_field_order()) from the py.dict returned by
%   marapendi.interop.simulink_bridge.diagnostics(), as a MATLAB column vector.

    names = state_scalar_field_order();
    v = zeros(numel(names), 1);
    for k = 1:numel(names)
        v(k) = double(diagDict{names{k}});
    end
end
