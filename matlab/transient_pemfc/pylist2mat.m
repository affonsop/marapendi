function v = pylist2mat(pyList)
%PYLIST2MAT Convert a Python list of floats to a MATLAB column double vector.
    v = double(py.array.array('d', pyList))';
end
