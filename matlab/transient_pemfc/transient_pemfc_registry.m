function out = transient_pemfc_registry(action, val)
%TRANSIENT_PEMFC_REGISTRY Persistent id->py.object store.
%   Level-2 MATLAB S-Functions can only persist numeric data in Dwork
%   vectors across calls, but the cached FuelCell is a py.object handle.
%   Each S-Function instance stores an integer id in Dwork(1) and looks the
%   actual object up here.

    persistent registry nextId
    if isempty(registry)
        registry = containers.Map('KeyType', 'double', 'ValueType', 'any');
        nextId = 1;
    end

    switch action
        case 'store'
            id = nextId;
            nextId = nextId + 1;
            registry(id) = val;
            out = id;
        case 'get'
            out = registry(val);
        case 'clear'
            if isKey(registry, val)
                remove(registry, val);
            end
            out = [];
    end
end
