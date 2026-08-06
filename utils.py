import cv2
import joblib
import numpy as np
import pickle      

def arr_caps(final_l, arr):
    ld = final_l-len(arr)
    if ld == 1:
        arr = np.append(arr, arr[-1])
    else:
        prefill = np.full(int(np.floor(ld/2)), arr[0])
        postfill = np.full(int(np.ceil(ld/2)), arr[-1])
        arr = np.concatenate([prefill, arr, postfill])
    return arr

def diff_step(arr, step, prepend = None, append = None, cap = False):
    diff = np.array(arr[step:].to_numpy()-arr[:-step].to_numpy())
    if not cap:
        if prepend is not None:
            diff = np.concatenate([np.atleast_1d(prepend), diff])
        if append is not None:
            diff = np.concatenate([diff, np.atleast_1d(append)])
    else:
        diff = arr_caps(len(arr), diff)
    return diff

def load_pkl(filepath):

    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
            return obj
        
    except(ModuleNotFoundError, pickle.UnpicklingError):
        try:
            with open(filepath, 'rb') as f:
                obj = joblib.load(f)
                return obj
        except Exception as e:
            print(f'File not loaded: {e}')
            return None
        
    except FileNotFoundError:
        print(f'File not found: {filepath}')
        return None

def trim_avi(avi_path, cutoff):

    dst = f'{avi_path[:-4]}_cut.avi'
    cap = cv2.VideoCapture(avi_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(dst, fourcc, fps, (w,h))

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        
        out.write(frame)

        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == cutoff:
            break

    cap.release()
    out.release()

def verify_len(*args):
    
    lens = {len(obj) for obj in args}
    if len(lens) == 1:
        return 1
    else:
        print('Lengths did not all match.'+('\n'*2))
        for obj in args:
            print(f'{obj}: {len(obj)}'+'\n')
        return 0 