import numpy as np
import re

def py_to_mat(pyObj):

    if isinstance(pyObj, dict):
        matObj = {}
        for k, v in pyObj.items():
            if re.match('\\d', str(k)[0]):
                matObj[f'k{k}'] = py_to_mat(v)
            else:
                matObj[k] = py_to_mat(v)
        return matObj
    elif isinstance(pyObj, list) or isinstance(pyObj, tuple):
        try:
            return np.array([py_to_mat(x) for x in pyObj], dtype=np.int16)
        except:
            return {f'elem{i}': py_to_mat(x) for i, x in enumerate(pyObj)}
    elif isinstance(pyObj, bytes):
        return np.frombuffer(pyObj, dtype=np.int16) # int16 for ephys data!
    elif isinstance(pyObj, (int, float, bool, np.integer, np.floating, np.bool_)):
        return pyObj
    elif isinstance(pyObj, np.ndarray):
        return pyObj
    elif isinstance(pyObj, str):
        return pyObj
    else:
        raise TypeError(f"Unsupported Python object type: {type(pyObj)}")
