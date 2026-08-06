import cv2
import h5py
import numpy as np
from tqdm import tqdm


def extract(avi_path):

    prep_h5_dst = f'{avi_path[:-4]}_prep.h5'
    fin_h5_dst = f'{avi_path[:-4]}_final.h5'
    unsort_dst = f'{avi_path[:-4]}_unsort.mat'
    cap = cv2.VideoCapture(avi_path)

    if not cap.isOpened():
        raise IOError("Cannot open video file: {avi}")

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


        with tqdm(total=fc, desc='Converting to .h5', unit='frame') as pbar:
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

    