import cv2
import joblib
import numpy as np
import os
from pathlib import Path
import pickle 
import shutil
import subprocess   
from tqdm import tqdm  

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

def file_copy(src, dst):
    verify = True
    try:
        subprocess.run(["robocopy", src, dst, "/Z", "/MT:8"])
    except Exception as e:
        print(f'Error when copying file {src} to {dst}: {e}')
        verify = False
    return verify

def file_del(path):
    verify = True
    try:
        os.remove(path)
    except Exception as e:
        print(f'Error removing file {path}: {e}')
        verify = False
    return verify

def folder_copy(src, dst):
    verify = True
    try:
        src = Path(src)
        dst = Path(dst)

        size = sum(f.stat().st_size for f in src.rglob('*') if f.is_file())

        with tqdm(total=size, unit='B', unit_scale=True, unit_divisor=1024,
                  desc=f'Copying {src.name} to local...') as pbar:
            def copy_by_file(srcp, dstp):
                
                shutil.copy2(srcp, dstp)
                pbar.update(srcp.stat().st_size)

                shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=copy_by_file)

    except Exception as e:
        print(f'copying folder {src} to {dst} failed: {e}')
        verify = False
    return verify

def folder_del(path):
    verify = True
    try: 
        shutil.rmtree(path)
    except Exception as e:
        print(f'failed to remove folder {path}: {e}')
        verify = False
    return verify

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