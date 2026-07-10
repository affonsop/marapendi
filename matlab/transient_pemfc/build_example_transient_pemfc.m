function build_example_transient_pemfc(pythonExe)
%BUILD_EXAMPLE_TRANSIENT_PEMFC Build example_transient_pemfc.slx, a runnable
%   demo of the TransientPEMFC block: a step change in current density
%   (10000 -> 20000 A/m^2 at t=100s) at otherwise fixed inlet GasFlowStates,
%   scoped on cell voltage, MEA temperature, heat release rate, and the
%   cathode outlet water content.
%
%   Requires TransientPEMFC.slx to already exist (run build_transient_block
%   first) — this script copies that block into a new top-level model
%   rather than rebuilding it.
%
%   build_example_transient_pemfc() uses the currently configured pyenv().
%   build_example_transient_pemfc(pythonExe) switches interpreter first.

    if nargin < 1
        pythonExe = '';
    end

    here = fileparts(mfilename('fullpath'));
    addpath(here);

    pyenv_setup(pythonExe);
    n_memb_mesh = 5;
    create_buses(n_memb_mesh);

    libModel = 'TransientPEMFC';
    libPath = fullfile(here, [libModel '.slx']);
    if ~isfile(libPath)
        error('build_example_transient_pemfc:missingLib', ...
            'TransientPEMFC.slx not found at %s -- run build_transient_block first.', libPath);
    end
    libWasLoaded = bdIsLoaded(libModel);
    if ~libWasLoaded
        load_system(libPath);
    end

    modelName = 'example_transient_pemfc';
    if bdIsLoaded(modelName)
        close_system(modelName, 0);
    end
    new_system(modelName);
    open_system(modelName);

    add_block([libModel '/TransientPEMFC'], [modelName '/TransientPEMFC'], ...
        'Position', [320 100 460 260]);

    % Override x0 for this example's actual pre-step operating point (the
    % block's built-in default x0 is for a different nominal condition) so
    % the run starts at steady state instead of settling for the first
    % ~100s before the current step even happens.
    set_param([modelName '/TransientPEMFC'], 'x0', ...
        '[0.9783731338587631, 0.367695832038551, 0.3881598008019823, 0.40967937883127536, 0.4323084437233552, 0.4561035990835201]');

    % ---- Inputs: step current density, fixed inlet GasFlowStates -----------
    add_block('simulink/Sources/Step', [modelName '/CurrentDensity'], ...
        'Time', '100', 'Before', '10000', 'After', '20000', ...
        'Position', [40 40 90 70]);
    add_block('built-in/Constant', [modelName '/CellTemperature'], ...
        'Value', '344.15', 'Position', [40 100 90 120]);

    % Reference operating point: SideConditions matching
    % examples/plot_01_polarization_curve.py, converted to the equivalent
    % inlet GasFlowStates (via GasFlowState.from_side_conditions) at the
    % POST-STEP current density (20000 A/m^2) so the inlet flow keeps
    % enough O2/H2 headroom for the full step -- sizing it for the
    % pre-step 10000 A/m^2 alone would starve the reactants after the step.
    caFlowStruct = struct('temperature', 344.15, 'pressure', 140000., ...
        'o2', 2.07285393e-7, 'n2', 7.79787907e-7, 'h2', 0., 'h2o', 6.48626920e-8, 'liquid', 0.);
    anFlowStruct = struct('temperature', 344.15, 'pressure', 190000., ...
        'o2', 0., 'n2', 0., 'h2', 3.62749438e-7, 'h2o', 3.83749167e-8, 'liquid', 0.);

    % Assign into the model's own Model Workspace (not the base workspace)
    % so the values are saved inside the .slx and don't depend on a script
    % having been run first in whatever MATLAB session opens this model.
    modelWorkspace = get_param(modelName, 'ModelWorkspace');
    modelWorkspace.assignin('caFlowStruct', caFlowStruct);
    modelWorkspace.assignin('anFlowStruct', anFlowStruct);

    add_block('built-in/Constant', [modelName '/CaInlet'], ...
        'Value', 'caFlowStruct', 'OutDataTypeStr', 'Bus: GasFlowStateBus', ...
        'Position', [40 160 90 180]);
    add_block('built-in/Constant', [modelName '/AnInlet'], ...
        'Value', 'anFlowStruct', 'OutDataTypeStr', 'Bus: GasFlowStateBus', ...
        'Position', [40 200 90 220]);

    add_line(modelName, 'CaInlet/1', 'TransientPEMFC/1');
    add_line(modelName, 'AnInlet/1', 'TransientPEMFC/2');
    add_line(modelName, 'CurrentDensity/1', 'TransientPEMFC/3');
    add_line(modelName, 'CellTemperature/1', 'TransientPEMFC/4');

    % ---- Outputs: scope voltage, temperature, heat release, ca outlet H2O --
    add_block('simulink/Signal Routing/Bus Selector', [modelName '/DiagnosticsSelector'], ...
        'Position', [520 40 580 100]);
    set_param([modelName '/DiagnosticsSelector'], 'OutputSignals', 'cell_voltage,mea_temperature');
    add_line(modelName, 'TransientPEMFC/1', 'DiagnosticsSelector/1');

    add_block('simulink/Sinks/Scope', [modelName '/VoltageAndTemperatureScope'], ...
        'Position', [640 30 680 90], 'NumInputPorts', '2');
    add_line(modelName, 'DiagnosticsSelector/1', 'VoltageAndTemperatureScope/1');
    add_line(modelName, 'DiagnosticsSelector/2', 'VoltageAndTemperatureScope/2');

    add_block('simulink/Sinks/Scope', [modelName '/StateScope'], 'Position', [640 150 680 190]);
    add_line(modelName, 'TransientPEMFC/2', 'StateScope/1');

    add_block('simulink/Signal Routing/Bus Selector', [modelName '/CaOutletSelector'], ...
        'Position', [520 220 580 260]);
    set_param([modelName '/CaOutletSelector'], 'OutputSignals', 'h2o');
    add_line(modelName, 'TransientPEMFC/3', 'CaOutletSelector/1');

    add_block('simulink/Sinks/Scope', [modelName '/HeatReleaseAndCaOutletH2OScope'], ...
        'Position', [640 210 680 270], 'NumInputPorts', '2');
    add_line(modelName, 'TransientPEMFC/5', 'HeatReleaseAndCaOutletH2OScope/1');
    add_line(modelName, 'CaOutletSelector/1', 'HeatReleaseAndCaOutletH2OScope/2');

    set_param(modelName, 'StopTime', '300', 'SolverType', 'Variable-step', ...
        'Solver', 'ode15s', 'RelTol', '1e-3', ...
        'InitFcn', sprintf('pyenv_setup(); create_buses(%d);', n_memb_mesh));

    save_system(modelName, fullfile(here, [modelName '.slx']));
    fprintf('build_example_transient_pemfc: wrote %s\n', fullfile(here, [modelName '.slx']));

    if ~libWasLoaded
        close_system(libModel, 0);
    end
end
