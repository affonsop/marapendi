function pyenv_setup(pythonExe)
%PYENV_SETUP Point MATLAB's Python interface at the marapendi repo.
%
%   pyenv_setup() uses the currently configured pyenv() Python interpreter
%   (whatever `python` resolves to, or one already set via pyenv('Version', ...)).
%
%   pyenv_setup(pythonExe) additionally switches to the given interpreter,
%   e.g. pyenv_setup('/path/to/marapendi/.venv/bin/python').
%
%   Either way, this adds <repo>/src to Python's sys.path so
%   `import marapendi` resolves without installing the package, and warms
%   the import so failures surface here rather than inside the S-Function.

    if nargin >= 1 && ~isempty(pythonExe)
        pyenv('Version', pythonExe);
    end

    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(fileparts(thisFile)));  % matlab/transient_pemfc/.. -> matlab/.. -> repo root
    srcPath = fullfile(repoRoot, 'src');

    if count(py.sys.path, srcPath) == 0
        insert(py.sys.path, int32(0), srcPath);
    end

    py.importlib.import_module('marapendi.interop.simulink_bridge');

    fprintf('pyenv_setup: using %s, marapendi importable from %s\n', ...
        char(pyenv().Executable), srcPath);
end
