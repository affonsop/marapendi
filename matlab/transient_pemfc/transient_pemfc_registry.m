function out = transient_pemfc_registry(action, a, b)
%TRANSIENT_PEMFC_REGISTRY Persistent id->value store (py.object handles,
%   cached output structs, ...).
%   Level-2 MATLAB S-Functions can only persist numeric data in Dwork
%   vectors across calls, but the cached FuelCell is a py.object handle (and
%   the major-time-step output cache is a MATLAB struct) — neither fits in a
%   Dwork. Each S-Function instance stores an integer id in Dwork(1) and
%   looks the actual value up here.
%
%   id = transient_pemfc_registry('store', value)   create a new entry
%   value = transient_pemfc_registry('get', id)      read it back
%   transient_pemfc_registry('set', id, value)       overwrite it
%   transient_pemfc_registry('clear', id)            remove it

    persistent registry nextId
    if isempty(registry)
        registry = containers.Map('KeyType', 'double', 'ValueType', 'any');
        nextId = 1;
    end

    switch action
        case 'store'
            id = nextId;
            nextId = nextId + 1;
            registry(id) = a;
            out = id;
        case 'get'
            out = registry(a);
        case 'set'
            registry(a) = b;
            out = [];
        case 'clear'
            if isKey(registry, a)
                remove(registry, a);
            end
            out = [];
    end
end
