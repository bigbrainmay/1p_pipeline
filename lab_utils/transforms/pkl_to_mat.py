import sys

def pkl_to_mat(src, dst):
    from scipy.io import savemat
    from lab_utils import load_pkl, py_to_mat

    pyObject = load_pkl(src)
    matObject = py_to_mat(pyObject)
    if not isinstance(matObject, dict):
        matObject = {'matObject': matObject}
    savemat(dst, matObject)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise ValueError('Provide source and destination paths as arguments.')
    
    s = sys.argv[1]
    d = sys.argv[2]
    print(s)
    print(d)
    pkl_to_mat(s, d)