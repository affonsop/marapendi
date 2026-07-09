function pyObj = call_python_builder(dottedName)
%CALL_PYTHON_BUILDER Call a zero-argument Python function given as
%   'package.module.function' and return its result, e.g.
%   call_python_builder('marapendi.interop.simulink_bridge.build_default_cell').

    parts = strsplit(dottedName, '.');
    modName = strjoin(parts(1:end-1), '.');
    funcName = parts{end};
    mod = py.importlib.import_module(modName);
    fn = py.getattr(mod, funcName);
    pyObj = fn();
end
