function transient_pemfc_sfun(block)
%TRANSIENT_PEMFC_SFUN Level-2 MATLAB S-Function wrapping
%   marapendi.models.base.transient.TransientModel via
%   marapendi.interop.simulink_bridge (called through MATLAB's py. interface).
%
%   Continuous states: [T_mea_norm; lambda_1_norm; ...; lambda_N_norm], the
%   same normalised ODE state TransientModel.f_transient integrates.
%
%   Ports are plain vectors (Level-2 MATLAB S-Functions cannot reliably
%   declare bus-typed ports from M-code across MATLAB releases); the
%   TransientPEMFC masked subsystem built by build_transient_block.m wraps
%   this core with a Bus Selector on the way in and a Bus Creator on the way
%   out so the block's external interface is still bus-in/bus-out.
%
%   Input port 1  : flattened CellConditions, width 24, order = cond_field_order()
%   Output port 1 : flattened CellState scalars, width 26, order = state_scalar_field_order()
%   Output port 2 : membrane water-content profile, width n_memb_mesh
%   Output port 3 : raw ODE state vector, width n_memb_mesh + 1
%
%   Mask (DialogPrm) parameters, in order:
%     1. n_memb_mesh      - membrane finite-volume node count (must match the
%                            bus definitions from create_buses.m)
%     2. cellBuilderExpr  - dotted Python path to a zero-arg function building
%                            the FuelCell, e.g.
%                            'marapendi.interop.simulink_bridge.build_default_cell'
%     3. x0               - initial ODE state, numeric vector length n_memb_mesh+1
%                            (compute one with marapendi.interop.simulink_bridge.
%                            initial_state from Python, or via py. from MATLAB)
%
%   Limitation: every derivative evaluation, and every *major-time-step*
%   output evaluation, round-trips into Python and holds the GIL, so this
%   block does not support Rapid Accelerator or multicore execution.
%   Outputs are only recomputed on major time steps (Simulink's standard
%   pattern for expensive output blocks, via block.IsMajorTimeStep) — minor
%   steps (used internally by the variable-step solver for error estimation,
%   not part of the reported trajectory) reuse the last computed CellState
%   without calling Python. See matlab/transient_pemfc/README.md.

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

    block.NumInputPorts  = 1;
    block.NumOutputPorts = 3;

    block.SetPreCompInpPortInfoToDynamic;
    block.SetPreCompOutPortInfoToDynamic;

    block.InputPort(1).Dimensions        = numel(cond_field_order());
    block.InputPort(1).DatatypeID        = 0;
    block.InputPort(1).Complexity        = 'Real';
    block.InputPort(1).DirectFeedthrough = true;

    block.OutputPort(1).Dimensions       = numel(state_scalar_field_order());
    block.OutputPort(1).DatatypeID       = 0;
    block.OutputPort(1).Complexity       = 'Real';

    block.OutputPort(2).Dimensions       = n_memb_mesh;
    block.OutputPort(2).DatatypeID       = 0;
    block.OutputPort(2).Complexity       = 'Real';

    block.OutputPort(3).Dimensions       = nStates;
    block.OutputPort(3).DatatypeID       = 0;
    block.OutputPort(3).Complexity       = 'Real';

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
        'scalarVec', [], ...
        'profileVec', []);
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
    condDict = vec2conddict(block.InputPort(1).Data);
    dxdt = py.marapendi.interop.simulink_bridge.derivative( ...
        entry.pyCell, int32(n), block.CurrentTime, py.list(x(:)'), condDict);
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
        condDict = vec2conddict(block.InputPort(1).Data);
        diagDict = py.marapendi.interop.simulink_bridge.diagnostics( ...
            entry.pyCell, int32(n), block.CurrentTime, py.list(x(:)'), condDict);
        entry.scalarVec = diagdict2scalarvec(diagDict);
        entry.profileVec = pylist2mat(diagDict{'membrane_water_content_profile'});
        transient_pemfc_registry('set', id, entry);
    end

    block.OutputPort(1).Data = entry.scalarVec;
    block.OutputPort(2).Data = entry.profileVec;
    block.OutputPort(3).Data = x;
end

function Terminate(block)
    transient_pemfc_registry('clear', block.Dwork(1).Data);
end
