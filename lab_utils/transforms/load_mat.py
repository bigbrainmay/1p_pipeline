import os
from pathlib import Path
import pickle

def load_mat(file, pkl = False, dst = None):
    import matlab.engine
    from lab_utils import matdouble_to_numpy
    fp = os.path.abspath(__file__)
    pp = os.path.dirname(fp)
    engine = matlab.engine.start_matlab()
    engine.addpath(pp, nargout=0)
    mat = engine.load(file)
    py = engine.mat_to_py(mat)
    py = matdouble_to_numpy(py)
    
    if pkl:
        if not dst:
            filepath = Path(file)
            filename = os.path.splitext(os.path.basename(file))[0]
            folder = filepath.parent
            dst = os.path.join(folder, f'{filename}_py.pkl')
            
        with open(dst, 'wb') as f:
            pickle.dump(py, f)
        engine.quit()
        return
        
    else:
        engine.quit()
        return py
