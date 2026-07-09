function build_transient_block(pythonExe)
%BUILD_TRANSIENT_BLOCK Programmatically build TransientPEMFC.slx.
%
%   build_transient_block() uses the currently configured pyenv().
%   build_transient_block(pythonExe) switches to the given interpreter first
%   (e.g. build_transient_block('/path/to/marapendi/.venv/bin/python')).
%
%   Produces matlab/transient_pemfc/TransientPEMFC.slx: a masked subsystem
%   with a CellConditionsBus inport and two outports (CellStateBus, the raw
%   ODE state vector x). Internally: Bus Selector -> Mux -> S-Function
%   (transient_pemfc_sfun, plain vector I/O) -> Demux -> Bus Creator. Level-2
%   MATLAB S-Functions cannot reliably declare bus-typed ports from M-code
%   across MATLAB releases, so the bus/vector conversion is done with these
%   standard blocks instead of on the S-Function's own ports.

    if nargin < 1
        pythonExe = '';
    end

    here = fileparts(mfilename('fullpath'));
    addpath(here);

    pyenv_setup(pythonExe);

    n_memb_mesh = 5;
    create_buses(n_memb_mesh);

    cellBuilderExpr = 'marapendi.interop.simulink_bridge.build_default_cell';

    % Nominal reference operating point used only to compute a default x0 for
    % the mask; the block itself uses whatever CellConditionsBus is wired in.
    nominalCond = struct( ...
        'current_density', 10000., 'cell_temperature', 344.15, ...
        'ca', local_side(344.15, 140000., 140000., 0.21, 0.0, 0.265, 1.6), ...
        'an', local_side(344.15, 190000., 190000., 0.0,  1.0, 0.558, 1.4));

    pyCell = call_python_builder(cellBuilderExpr);
    condDict = busconditions2dict(nominalCond);
    x0 = pylist2mat(py.marapendi.interop.simulink_bridge.initial_state( ...
        pyCell, int32(n_memb_mesh), condDict))';

    % The S-Function block's 'Parameters' expression is evaluated in the base
    % workspace, not this function's local workspace — publish there too.
    assignin('base', 'n_memb_mesh', n_memb_mesh);
    assignin('base', 'cellBuilderExpr', cellBuilderExpr);
    assignin('base', 'x0', x0);

    modelName = 'TransientPEMFC';
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
    new_system(modelName);
    open_system(modelName);

    % ---- CellConditions -> flat vector (Bus Selector + Mux) ---------------
    add_block('built-in/Inport', [modelName '/CellConditions'], ...
        'Position', [30 150 70 170], 'OutDataTypeStr', 'Bus: CellConditionsBus');

    condNames = cond_field_order();
    dotted = condNames;
    for i = 1:numel(dotted)
        if startsWith(dotted{i}, 'ca_')
            dotted{i} = ['ca.' dotted{i}(4:end)];
        elseif startsWith(dotted{i}, 'an_')
            dotted{i} = ['an.' dotted{i}(4:end)];
        end
    end
    busSel = [modelName '/BusSelector'];
    add_block('simulink/Signal Routing/Bus Selector', busSel, 'Position', [150 20 220 500]);
    set_param(busSel, 'OutputSignals', strjoin(dotted, ','));
    add_line(modelName, 'CellConditions/1', 'BusSelector/1');

    mux = [modelName '/Mux'];
    add_block('simulink/Signal Routing/Mux', mux, 'Inputs', num2str(numel(condNames)), ...
        'Position', [280 20 330 500]);
    for i = 1:numel(condNames)
        add_line(modelName, sprintf('BusSelector/%d', i), sprintf('Mux/%d', i));
    end

    % ---- Core S-Function ----------------------------------------------------
    sfBlock = [modelName '/TransientPEMFC_core'];
    add_block('simulink/User-Defined Functions/Level-2 MATLAB S-Function', sfBlock, ...
        'FunctionName', 'transient_pemfc_sfun', ...
        'Parameters', 'n_memb_mesh, cellBuilderExpr, x0', ...
        'Position', [380 200 520 320]);
    add_line(modelName, 'Mux/1', 'TransientPEMFC_core/1');

    % ---- flat vector -> CellState (Demux + Bus Creator) --------------------
    stateNames = state_scalar_field_order();
    demux = [modelName '/Demux'];
    add_block('simulink/Signal Routing/Demux', demux, 'Outputs', num2str(numel(stateNames)), ...
        'Position', [580 20 620 500]);
    add_line(modelName, 'TransientPEMFC_core/1', 'Demux/1');

    busCreator = [modelName '/BusCreator'];
    add_block('simulink/Signal Routing/Bus Creator', busCreator, ...
        'Inputs', num2str(numel(stateNames) + 1), 'Position', [680 20 760 520]);
    set_param(busCreator, 'UseBusObject', 'on', 'BusObject', 'CellStateBus', 'NonVirtualBus', 'on');
    for i = 1:numel(stateNames)
        h = add_line(modelName, sprintf('Demux/%d', i), sprintf('BusCreator/%d', i));
        set_param(h, 'Name', stateNames{i});
    end
    h = add_line(modelName, 'TransientPEMFC_core/2', sprintf('BusCreator/%d', numel(stateNames) + 1));
    set_param(h, 'Name', 'membrane_water_content_profile');

    add_block('built-in/Outport', [modelName '/CellState'], 'Position', [820 260 860 280]);
    add_line(modelName, 'BusCreator/1', 'CellState/1');

    add_block('built-in/Outport', [modelName '/x'], 'Position', [580 550 620 570]);
    add_line(modelName, 'TransientPEMFC_core/3', 'x/1');

    % ---- Group into one masked "TransientPEMFC" block ----------------------
    blocksToGroup = {[modelName '/CellConditions'], [modelName '/BusSelector'], [modelName '/Mux'], ...
                      [modelName '/TransientPEMFC_core'], [modelName '/Demux'], [modelName '/BusCreator'], ...
                      [modelName '/CellState'], [modelName '/x']};
    handles = cell2mat(get_param(blocksToGroup, 'Handle'));
    Simulink.BlockDiagram.createSubsystem(handles, 'Name', 'TransientPEMFC');
    subBlockPath = [modelName '/TransientPEMFC'];

    mask = Simulink.Mask.create(subBlockPath);
    mask.Type = 'TransientPEMFC';
    mask.Description = sprintf(['Encapsulates marapendi.models.base.transient.TransientModel\n' ...
        '(coupled MEA temperature / membrane water-content ODE) via a Python-calling S-Function.\n' ...
        'Edit the physics in the marapendi Python package; this block always calls the live model.']);

    addMaskParam(mask, 'n_memb_mesh', 'edit', 'Membrane mesh nodes (n_memb_mesh)', num2str(n_memb_mesh));
    addMaskParam(mask, 'cellBuilderExpr', 'edit', 'Python cell-builder function (dotted path)', ['''' cellBuilderExpr '''']);
    addMaskParam(mask, 'x0', 'edit', 'Initial ODE state x0 (1 x n_memb_mesh+1)', mat2str(x0));

    set_param(modelName, 'InitFcn', sprintf('pyenv_setup(); create_buses(%d);', n_memb_mesh));

    save_system(modelName, fullfile(here, [modelName '.slx']));
    fprintf('build_transient_block: wrote %s\n', fullfile(here, [modelName '.slx']));
end

function addMaskParam(mask, name, style, prompt, value)
    p = mask.addParameter('Type', style, 'Name', name, 'Prompt', prompt);
    p.Value = value;
end

function s = local_side(T, pin, pout, xo2, xh2, rh, stoich)
    s = struct( ...
        'inlet_temperature', T, 'inlet_pressure', pin, 'outlet_pressure', pout, ...
        'dry_o2_mole_fraction', xo2, 'dry_h2_mole_fraction', xh2, ...
        'inlet_relative_humidity', rh, 'stoichiometry', stoich, ...
        'inlet_liquid_saturation', 0., 'inlet_liquid_flow_rate', 0., ...
        'inlet_gas_flow_rate', 0., 'minimum_current_density_for_stoich', 0.);
end
