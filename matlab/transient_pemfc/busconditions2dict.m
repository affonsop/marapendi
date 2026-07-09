function d = busconditions2dict(condBus)
%BUSCONDITIONS2DICT Flatten a CellConditionsBus struct into the py.dict shape
%   expected by marapendi.interop.simulink_bridge (current_density,
%   cell_temperature, ca_<field>, an_<field>).

    sideFields = {
        'inlet_temperature', 'inlet_pressure', 'outlet_pressure', ...
        'dry_o2_mole_fraction', 'dry_h2_mole_fraction', ...
        'inlet_relative_humidity', 'stoichiometry', ...
        'inlet_liquid_saturation', 'inlet_liquid_flow_rate', ...
        'inlet_gas_flow_rate', 'minimum_current_density_for_stoich'
    };

    args = {'current_density', double(condBus.current_density), ...
            'cell_temperature', double(condBus.cell_temperature)};

    for prefix = {'ca', 'an'}
        p = prefix{1};
        for k = 1:numel(sideFields)
            f = sideFields{k};
            args{end+1} = [p '_' f]; %#ok<AGROW>
            args{end+1} = double(condBus.(p).(f)); %#ok<AGROW>
        end
    end

    d = py.dict(pyargs(args{:}));
end
