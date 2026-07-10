function build_transient_block(pythonExe)
%BUILD_TRANSIENT_BLOCK Programmatically build TransientPEMFC.slx.
%
%   build_transient_block() uses the currently configured pyenv().
%   build_transient_block(pythonExe) switches to the given interpreter first.
%
%   Produces matlab/transient_pemfc/TransientPEMFC.slx: a masked subsystem
%   driven directly by inlet GasFlowStates (cathode + anode) plus current
%   density and cell temperature -- the natural inputs for a system-level
%   model where upstream compressor/humidifier/coolant models already
%   produce flow states, not a stoichiometry spec. See
%   matlab/transient_pemfc/README.md for the wiring pattern this uses
%   (Bus Selector -> Mux -> S-Function -> Demux -> Bus Creator, since
%   Level-2 MATLAB S-Functions cannot reliably declare bus-typed ports
%   from M-code).
%
%   Ports:
%     Inputs  - CaInlet, AnInlet (bus, GasFlowStateBus), CurrentDensity,
%               CellTemperature (scalar)
%     Outputs - CellState (bus, CellStateBus), CaOutlet, AnOutlet (bus,
%               GasFlowStateBus), HeatRelease, CellVoltage (scalar), x (raw
%               ODE state vector)

    if nargin < 1
        pythonExe = '';
    end

    here = fileparts(mfilename('fullpath'));
    addpath(here);

    pyenv_setup(pythonExe);

    n_memb_mesh = 5;
    create_buses(n_memb_mesh);

    cellBuilderExpr = 'marapendi.interop.simulink_bridge.build_default_cell';

    % Nominal reference operating point used only to compute a default x0
    % for the mask; the block itself uses whatever GasFlowStates are wired in.
    pyCell = call_python_builder(cellBuilderExpr);
    caFlowStruct = struct('temperature', 344.15, 'pressure', 140000., ...
        'o2', 1.05e-7, 'n2', 3.90e-7, 'h2', 0., 'h2o', 3.25e-8, 'liquid', 0.);
    anFlowStruct = struct('temperature', 344.15, 'pressure', 190000., ...
        'o2', 0., 'n2', 0., 'h2', 1.82e-7, 'h2o', 1.92e-8, 'liquid', 0.);
    x0 = pylist2mat(py.marapendi.interop.simulink_bridge.cell_initial_state( ...
        pyCell, int32(n_memb_mesh), ...
        struct2flowdict(caFlowStruct), struct2flowdict(anFlowStruct), ...
        10000., 344.15))';

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

    flowNames = gasflow_field_order();

    % ---- ca/an inlet GasFlowState -> flat vectors (Bus Selector + Mux) -----
    add_block('built-in/Inport', [modelName '/CaInlet'], ...
        'Position', [30 50 70 70], 'OutDataTypeStr', 'Bus: GasFlowStateBus');
    add_block('built-in/Inport', [modelName '/AnInlet'], ...
        'Position', [30 200 70 220], 'OutDataTypeStr', 'Bus: GasFlowStateBus');

    caBusSel = [modelName '/CaBusSelector'];
    add_block('simulink/Signal Routing/Bus Selector', caBusSel, 'Position', [140 20 210 200]);
    set_param(caBusSel, 'OutputSignals', strjoin(flowNames, ','));
    add_line(modelName, 'CaInlet/1', 'CaBusSelector/1');

    anBusSel = [modelName '/AnBusSelector'];
    add_block('simulink/Signal Routing/Bus Selector', anBusSel, 'Position', [140 180 210 360]);
    set_param(anBusSel, 'OutputSignals', strjoin(flowNames, ','));
    add_line(modelName, 'AnInlet/1', 'AnBusSelector/1');

    caMux = [modelName '/CaMux'];
    add_block('simulink/Signal Routing/Mux', caMux, 'Inputs', num2str(numel(flowNames)), ...
        'Position', [260 20 300 200]);
    for i = 1:numel(flowNames)
        add_line(modelName, sprintf('CaBusSelector/%d', i), sprintf('CaMux/%d', i));
    end

    anMux = [modelName '/AnMux'];
    add_block('simulink/Signal Routing/Mux', anMux, 'Inputs', num2str(numel(flowNames)), ...
        'Position', [260 180 300 360]);
    for i = 1:numel(flowNames)
        add_line(modelName, sprintf('AnBusSelector/%d', i), sprintf('AnMux/%d', i));
    end

    add_block('built-in/Inport', [modelName '/CurrentDensity'], 'Position', [30 400 70 420]);
    add_block('built-in/Inport', [modelName '/CellTemperature'], 'Position', [30 450 70 470]);

    % ---- Core S-Function ----------------------------------------------------
    sfBlock = [modelName '/TransientPEMFC_core'];
    add_block('simulink/User-Defined Functions/Level-2 MATLAB S-Function', sfBlock, ...
        'FunctionName', 'transient_pemfc_sfun', ...
        'Parameters', 'n_memb_mesh, cellBuilderExpr, x0', ...
        'Position', [380 150 560 320]);
    add_line(modelName, 'CaMux/1', 'TransientPEMFC_core/1');
    add_line(modelName, 'AnMux/1', 'TransientPEMFC_core/2');
    add_line(modelName, 'CurrentDensity/1', 'TransientPEMFC_core/3');
    add_line(modelName, 'CellTemperature/1', 'TransientPEMFC_core/4');

    % ---- flat vector -> CellState (Demux + Bus Creator) --------------------
    stateNames = state_scalar_field_order();
    demux = [modelName '/StateDemux'];
    add_block('simulink/Signal Routing/Demux', demux, 'Outputs', num2str(numel(stateNames)), ...
        'Position', [620 20 660 500]);
    add_line(modelName, 'TransientPEMFC_core/1', 'StateDemux/1');

    busCreator = [modelName '/StateBusCreator'];
    add_block('simulink/Signal Routing/Bus Creator', busCreator, ...
        'Inputs', num2str(numel(stateNames) + 1), 'Position', [720 20 800 520]);
    set_param(busCreator, 'UseBusObject', 'on', 'BusObject', 'CellStateBus', 'NonVirtualBus', 'on');
    for i = 1:numel(stateNames)
        h = add_line(modelName, sprintf('StateDemux/%d', i), sprintf('StateBusCreator/%d', i));
        set_param(h, 'Name', stateNames{i});
    end
    h = add_line(modelName, 'TransientPEMFC_core/2', sprintf('StateBusCreator/%d', numel(stateNames) + 1));
    set_param(h, 'Name', 'membrane_water_content_profile');

    add_block('built-in/Outport', [modelName '/CellState'], 'Position', [860 260 900 280]);
    add_line(modelName, 'StateBusCreator/1', 'CellState/1');

    add_block('built-in/Outport', [modelName '/x'], 'Position', [620 550 660 570]);
    add_line(modelName, 'TransientPEMFC_core/3', 'x/1');

    % ---- ca/an outlet flat vectors -> GasFlowStateBus (Demux + Bus Creator) --
    add_gasflow_output(modelName, 'CaOutlet', 'TransientPEMFC_core/4', flowNames, [620 620 660 660], [720 610 800 780], [860 690 900 710]);
    add_gasflow_output(modelName, 'AnOutlet', 'TransientPEMFC_core/5', flowNames, [620 800 660 840], [720 790 800 960], [860 870 900 890]);

    add_block('built-in/Outport', [modelName '/HeatRelease'], 'Position', [620 950 660 970]);
    add_line(modelName, 'TransientPEMFC_core/6', 'HeatRelease/1');

    add_block('built-in/Outport', [modelName '/CellVoltage'], 'Position', [620 990 660 1010]);
    add_line(modelName, 'TransientPEMFC_core/7', 'CellVoltage/1');

    % ---- Group into one masked "TransientPEMFC" block ------------------------
    blocksToGroup = {[modelName '/CaInlet'], [modelName '/AnInlet'], ...
                      [modelName '/CurrentDensity'], [modelName '/CellTemperature'], ...
                      [modelName '/CaBusSelector'], [modelName '/AnBusSelector'], ...
                      [modelName '/CaMux'], [modelName '/AnMux'], ...
                      [modelName '/TransientPEMFC_core'], ...
                      [modelName '/StateDemux'], [modelName '/StateBusCreator'], [modelName '/CellState'], ...
                      [modelName '/x'], ...
                      [modelName '/CaOutlet_Demux'], [modelName '/CaOutlet_BusCreator'], [modelName '/CaOutlet'], ...
                      [modelName '/AnOutlet_Demux'], [modelName '/AnOutlet_BusCreator'], [modelName '/AnOutlet'], ...
                      [modelName '/HeatRelease'], [modelName '/CellVoltage']};
    handles = cell2mat(get_param(blocksToGroup, 'Handle'));
    Simulink.BlockDiagram.createSubsystem(handles, 'Name', 'TransientPEMFC');
    subBlockPath = [modelName '/TransientPEMFC'];

    mask = Simulink.Mask.create(subBlockPath);
    mask.Type = 'TransientPEMFC';
    mask.Description = sprintf([ ...
        'Encapsulates marapendi.models.base.transient.TransientModel driven\n' ...
        'directly by inlet GasFlowStates (cathode + anode) + current density\n' ...
        '+ cell temperature, via a Python-calling S-Function. Edit the physics\n' ...
        'in the marapendi Python package; this block always calls the live model.']);

    addMaskParam(mask, 'n_memb_mesh', 'edit', 'Membrane mesh nodes (n_memb_mesh)', num2str(n_memb_mesh));
    addMaskParam(mask, 'cellBuilderExpr', 'edit', 'Python cell-builder function (dotted path)', ['''' cellBuilderExpr '''']);
    addMaskParam(mask, 'x0', 'edit', 'Initial ODE state x0 (1 x n_memb_mesh+1)', mat2str(x0));

    set_param(modelName, 'InitFcn', sprintf('pyenv_setup(); create_buses(%d);', n_memb_mesh));

    save_system(modelName, fullfile(here, [modelName '.slx']));
    fprintf('build_transient_block: wrote %s\n', fullfile(here, [modelName '.slx']));
end

function add_gasflow_output(modelName, outportName, srcPort, flowNames, demuxPos, busCreatorPos, outportPos)
    demux = [modelName '/' outportName '_Demux'];
    add_block('simulink/Signal Routing/Demux', demux, 'Outputs', num2str(numel(flowNames)), 'Position', demuxPos);
    add_line(modelName, srcPort, [outportName '_Demux/1']);

    busCreator = [modelName '/' outportName '_BusCreator'];
    add_block('simulink/Signal Routing/Bus Creator', busCreator, ...
        'Inputs', num2str(numel(flowNames)), 'Position', busCreatorPos);
    set_param(busCreator, 'UseBusObject', 'on', 'BusObject', 'GasFlowStateBus', 'NonVirtualBus', 'on');
    for i = 1:numel(flowNames)
        h = add_line(modelName, sprintf('%s_Demux/%d', outportName, i), sprintf('%s_BusCreator/%d', outportName, i));
        set_param(h, 'Name', flowNames{i});
    end

    add_block('built-in/Outport', [modelName '/' outportName], 'Position', outportPos);
    add_line(modelName, [outportName '_BusCreator/1'], [outportName '/1']);
end

function addMaskParam(mask, name, style, prompt, value)
    p = mask.addParameter('Type', style, 'Name', name, 'Prompt', prompt);
    p.Value = value;
end

function d = struct2flowdict(s)
    d = py.dict(pyargs( ...
        'temperature', s.temperature, 'pressure', s.pressure, ...
        'o2', s.o2, 'n2', s.n2, 'h2', s.h2, 'h2o', s.h2o, 'liquid', s.liquid));
end
