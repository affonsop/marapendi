function pyObj = call_python_builder(builderExpr)
%CALL_PYTHON_BUILDER Build the FuelCell used by the TransientPEMFC block.
%
%   Two forms of BUILDEREXPR are supported, distinguished by whether it
%   contains a '.':
%
%   1. A dotted Python path 'package.module.function' to a zero-argument
%      Python function returning a FuelCell, e.g.
%      'marapendi.interop.simulink_bridge.build_default_cell'.
%
%   2. The name of a MATLAB function on the path (no '.') that takes no
%      arguments and returns a struct of cell parameters -- see
%      cell_params_template.m for the full set of fields/defaults/grouping.
%      Copy that file (e.g. to my_cell_params.m), edit the values, and pass
%      'my_cell_params' here. The struct is converted to a nested py.dict
%      (matstruct2pydict.m) and passed to
%      marapendi.interop.simulink_bridge.build_cell_from_params().

    if contains(builderExpr, '.')
        parts = strsplit(builderExpr, '.');
        modName = strjoin(parts(1:end-1), '.');
        funcName = parts{end};
        mod = py.importlib.import_module(modName);
        fn = py.getattr(mod, funcName);
        pyObj = fn();
    else
        params = feval(builderExpr);
        pyObj = py.marapendi.interop.simulink_bridge.build_cell_from_params( ...
            matstruct2pydict(params));
    end
end
