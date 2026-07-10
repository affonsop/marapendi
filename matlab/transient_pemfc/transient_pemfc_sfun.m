function transient_pemfc_sfun(block)
%TRANSIENT_PEMFC_SFUN Level-2 MATLAB S-Function wrapping
%   marapendi.models.base.transient.TransientModel via
%   marapendi.interop.simulink_bridge (called through MATLAB's py. interface).
%
%   Driven directly by inlet GasFlowStates (cathode + anode) plus current
%   density and cell temperature — the natural inputs for a system-level
%   model where upstream compressor/humidifier/coolant models already
%   produce flow states, not a stoichiometry spec.
%
%   Continuous states: [T_mea_norm; lambda_1_norm; ...; lambda_N_norm], the
%   same normalised ODE state TransientModel.f_transient integrates.
%
%   Ports are plain vectors (Level-2 MATLAB S-Functions cannot reliably
%   declare bus-typed ports from M-code across MATLAB releases); the
%   TransientPEMFC masked subsystem built by build_transient_block.m wraps
%   this core with Bus Selectors on the way in and Bus Creators on the way
%   out so the block's external interface is bus-in/bus-out for the
%   GasFlowStates and the CellState.
%
%   Input port 1  : ca inlet GasFlowState, width 7, order = gasflow_field_order()
%   Input port 2  : an inlet GasFlowState, width 7, order = gasflow_field_order()
%   Input port 3  : current density (A/m^2), width 1
%   Input port 4  : cell temperature (K), width 1
%   Output port 1 : flattened CellState scalars, width 27, order = state_scalar_field_order()
%   Output port 2 : membrane water-content profile, width n_memb_mesh
%   Output port 3 : raw ODE state vector, width n_memb_mesh + 1
%   Output port 4 : ca outlet GasFlowState, width 7, order = gasflow_field_order()
%   Output port 5 : an outlet GasFlowState, width 7, order = gasflow_field_order()
%   Output port 6 : heat release rate (W/m^2), width 1
%   Output port 7 : cell voltage (V), width 1 (duplicates CellState's
%                   cell_voltage, exposed as its own port for convenience)
%
%   Mask (DialogPrm) parameters, in order:
%     1. n_memb_mesh      - membrane finite-volume node count (must match the
%                            bus definitions from create_buses.m)
%     2. cellBuilderExpr  - dotted Python path to a zero-arg function building
%                            the FuelCell, e.g.
%                            'marapendi.interop.simulink_bridge.build_default_cell'
%     3. x0               - initial ODE state, numeric vector length n_memb_mesh+1
%                            (compute one with marapendi.interop.simulink_bridge.
%                            cell_initial_state from Python, or via py. from MATLAB)
%
%   Limitation: every derivative evaluation, and every *major-time-step*
%   output evaluation, round-trips into Python and holds the GIL, so this
%   block does not support Rapid Accelerator or multicore execution. Outputs
%   are only recomputed on major time steps (see matlab/transient_pemfc/README.md
%   for why, and the measured effect on Python call counts).

    setup(block);
end

function setup(block)
    block.NumDialogPrms = 3;
    block.DialogPrmsTunable = {'Nontunable', 'Nontunable', 'Nontunable'};

    n_memb_mesh = double(block.DialogPrm(1).Data);
    x0 = double(block.DialogPrm(3).Data);
    nStates = n_memb_mesh + 1;
    if numel(x0) ~= nStates
        error('transient_pemfc_sfun:x0', ...
            'x0 must have n_memb_mesh + 1 = %d entries (got %d).', nStates, numel(x0));
    end

    block.NumInputPorts  = 4;
    block.NumOutputPorts = 7;

    block.SetPreCompInpPortInfoToDynamic;
    block.SetPreCompOutPortInfoToDynamic;

    nFlow = numel(gasflow_field_order());

    block.InputPort(1).Dimensions        = nFlow;   % ca inlet flow
    block.InputPort(1).DatatypeID        = 0;
    block.InputPort(1).Complexity        = 'Real';
    block.InputPort(1).DirectFeedthrough = true;

    block.InputPort(2).Dimensions        = nFlow;   % an inlet flow
    block.InputPort(2).DatatypeID        = 0;
    block.InputPort(2).Complexity        = 'Real';
    block.InputPort(2).DirectFeedthrough = true;

    block.InputPort(3).Dimensions        = 1;       % current density
    block.InputPort(3).DatatypeID        = 0;
    block.InputPort(3).Complexity        = 'Real';
    block.InputPort(3).DirectFeedthrough = true;

    block.InputPort(4).Dimensions        = 1;       % cell temperature
    block.InputPort(4).DatatypeID        = 0;
    block.InputPort(4).Complexity        = 'Real';
    block.InputPort(4).DirectFeedthrough = true;

    block.OutputPort(1).Dimensions       = numel(state_scalar_field_order());
    block.OutputPort(1).DatatypeID       = 0;
    block.OutputPort(1).Complexity       = 'Real';

    block.OutputPort(2).Dimensions       = n_memb_mesh;
    block.OutputPort(2).DatatypeID       = 0;
    block.OutputPort(2).Complexity       = 'Real';

    block.OutputPort(3).Dimensions       = nStates;
    block.OutputPort(3).DatatypeID       = 0;
    block.OutputPort(3).Complexity       = 'Real';

    block.OutputPort(4).Dimensions       = nFlow;   % ca outlet flow
    block.OutputPort(4).DatatypeID       = 0;
    block.OutputPort(4).Complexity       = 'Real';

    block.OutputPort(5).Dimensions       = nFlow;   % an outlet flow
    block.OutputPort(5).DatatypeID       = 0;
    block.OutputPort(5).Complexity       = 'Real';

    block.OutputPort(6).Dimensions       = 1;       % heat release rate
    block.OutputPort(6).DatatypeID       = 0;
    block.OutputPort(6).Complexity       = 'Real';

    block.OutputPort(7).Dimensions       = 1;       % cell voltage
    block.OutputPort(7).DatatypeID       = 0;
    block.OutputPort(7).Complexity       = 'Real';

    block.NumContStates = nStates;

    block.SampleTimes = [0 0];

    block.SetAccelRunOnTLC(false);
    block.SimStateCompliance = 'DefaultSimState';

    block.RegBlockMethod('PostPropagationSetup', @PostPropagationSetup);
    block.RegBlockMethod('InitializeConditions',  @InitializeConditions);
    block.RegBlockMethod('Start',                 @Start);
    block.RegBlockMethod('Outputs',                @Outputs);
    block.RegBlockMethod('Derivatives',             @Derivatives);
    block.RegBlockMethod('Terminate',               @Terminate);
end

function PostPropagationSetup(block)
    block.NumDworks = 1;
    block.Dwork(1).Name = 'pyCellId';
    block.Dwork(1).Dimensions = 1;
    block.Dwork(1).DatatypeID = 0;
    block.Dwork(1).Complexity = 'Real';
    block.Dwork(1).UsedAsDiscState = false;
end

function Start(block)
    cellBuilderExpr = char(block.DialogPrm(2).Data);
    entry = struct( ...
        'pyCell', call_python_builder(cellBuilderExpr), ...
        'scalarVec', [], 'profileVec', [], 'caOutletVec', [], 'anOutletVec', [], ...
        'heatRelease', 0, 'cellVoltage', 0);
    block.Dwork(1).Data = transient_pemfc_registry('store', entry);
end

function InitializeConditions(block)
    x0 = double(block.DialogPrm(3).Data);
    block.ContStates.Data = x0(:);
end

function Derivatives(block)
    n = double(block.DialogPrm(1).Data);
    entry = transient_pemfc_registry('get', block.Dwork(1).Data);
    x = block.ContStates.Data;
    caFlow = vec2flowdict(block.InputPort(1).Data);
    anFlow = vec2flowdict(block.InputPort(2).Data);
    currentDensity = block.InputPort(3).Data;
    cellTemperature = block.InputPort(4).Data;
    dxdt = py.marapendi.interop.simulink_bridge.cell_derivative( ...
        entry.pyCell, int32(n), block.CurrentTime, py.list(x(:)'), ...
        caFlow, anFlow, currentDensity, cellTemperature);
    block.Derivatives.Data = pylist2mat(dxdt);
end

function Outputs(block)
    id = block.Dwork(1).Data;
    entry = transient_pemfc_registry('get', id);
    x = block.ContStates.Data;

    if block.IsMajorTimeStep || isempty(entry.scalarVec)
        % Minor time steps (used internally by the variable-step solver for
        % error estimation/interpolation) are not part of the reported
        % trajectory, so only pay the Python round trip on major steps.
        n = double(block.DialogPrm(1).Data);
        caFlow = vec2flowdict(block.InputPort(1).Data);
        anFlow = vec2flowdict(block.InputPort(2).Data);
        currentDensity = block.InputPort(3).Data;
        cellTemperature = block.InputPort(4).Data;
        diagDict = py.marapendi.interop.simulink_bridge.cell_diagnostics( ...
            entry.pyCell, int32(n), block.CurrentTime, py.list(x(:)'), ...
            caFlow, anFlow, currentDensity, cellTemperature);
        entry.scalarVec = diagdict2scalarvec(diagDict);
        entry.profileVec = pylist2mat(diagDict{'membrane_water_content_profile'});
        entry.caOutletVec = diagdict2flowvec(diagDict, 'ca_outlet');
        entry.anOutletVec = diagdict2flowvec(diagDict, 'an_outlet');
        entry.heatRelease = double(diagDict{'heat_release'});
        entry.cellVoltage = double(diagDict{'cell_voltage'});
        transient_pemfc_registry('set', id, entry);
    end

    block.OutputPort(1).Data = entry.scalarVec;
    block.OutputPort(2).Data = entry.profileVec;
    block.OutputPort(3).Data = x;
    block.OutputPort(4).Data = entry.caOutletVec;
    block.OutputPort(5).Data = entry.anOutletVec;
    block.OutputPort(6).Data = entry.heatRelease;
    block.OutputPort(7).Data = entry.cellVoltage;
end

function Terminate(block)
    transient_pemfc_registry('clear', block.Dwork(1).Data);
end
