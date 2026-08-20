import cv2
import h5py
import numpy as np
import os
from tqdm import tqdm

import matlab.engine

from config import user, wd
from utils import file_copy, file_del

def acquire_cells(sess_path, sess_head):

    avi_path = os.path.join(sess_path, f'{sess_head}_cam0.avi')
    prep_h5_dst = f'{avi_path[:-9]}_prep.h5'
    fin_h5_dst = f'{avi_path[:-9]}_final.h5'
    unsort_mat_dst = f'{avi_path[:-9]}_unsorted.mat'
    final_dst = f'{avi_path[:-9]}_cells.csv'
    
    cap = cv2.VideoCapture(avi_path)

    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {avi_path}")

    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    with h5py.File(prep_h5_dst, "w") as h5f:
        dset = h5f.create_dataset(
            "data",
            shape=(fc, h, w),
            dtype="float32",
            chunks=(1, h, w),
        )

        frame_idx = 0


        with tqdm(total=fc, desc='Converting video to .h5...', unit='frame') as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame.ndim == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dset[frame_idx] = frame.astype(np.float32)
                frame_idx += 1

                pbar.update(1)
    cap.release()
    print('\n'*3+'.avi file converted to .h5!'+'\n'*3)

    print('Starting MATLAB engine...')
    eng = matlab.engine.start_engine()
    print('MATLAB engine started!'+'\n'*3)
    eng.addpath(eng.genpath(os.path.join(wd, 'matlab')))
    addons = os.path.join(fr'C:\Users\{user}\AppData\Roaming\MathWorks\MATLAB Add-Ons')
    actsort_path = os.path.join(addons, 'Apps', 'ActSort')
    extract_path = os.path.join(addons, 'Collections', 'EXTRACT-public')
    eng.addpath(eng.genpath(actsort_path), nargout=0)
    eng.addpath(eng.genpath(extract_path), nargout=0)

    print('Running EXTRACT...')
    extract_verify = eng.run_extract(prep_h5_dst, fin_h5_dst, unsort_mat_dst)
    print('EXTRACT successfully completed!'+'\n'*3)

    print(f'NOTE: About to launch ActSort. \
          Pretty please save the output in {sess_path} as  "{sess_head}_actsort.mat"'+'\n'*2)

    print('Have fun twin! Hope your data looks good!'+'\n')
    eng.ActSortApp(nargout=0)

    while input('Press [ENTER] when sorting has been completed.') != "":
        print('\n'+'That was NOT the [ENTER] key, be fr rn :|')

    print('\n'*2+'Pulling cell traces and saving to session folder...')
    actsort = eng.mat_to_py(os.path.join(sess_path, f'{sess_head}_actsort.mat'))
    labels = eng.mat_to_py(os.path.join(sess_path, f'{sess_head}_actsort_LABELS.mat'))

    traces = np.array(actsort['precomputedOutput']['traces'])
    
