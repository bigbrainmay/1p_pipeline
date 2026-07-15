import numpy as np

def matdouble_to_numpy(obj):
    import matlab.engine
    
    out = obj
    if isinstance(obj, dict):
        out = obj
        for k, v in obj.items():
            if isinstance(v, dict):
                out[k] = matdouble_to_numpy(v)
            elif isinstance(v, list):
                for i,e in enumerate(v):
                    out[k][i] = matdouble_to_numpy(e)
            elif isinstance(v, matlab.double):
                out[k] = np.array(v)

    elif isinstance(obj, list):
        for i,e in enumerate(obj):
            out[i] = matdouble_to_numpy(e)

    if isinstance(obj, matlab.double):
        out = np.array(obj)
        
    return out