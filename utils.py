import cv2
import joblib
import pickle      

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