import cv2
from tqdm import tqdm

def splice_avi(src, dst, stopframe, fourcc_s='XVID'):
    stopframe = int(stopframe)

    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise IOError(f"Could not open source video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*fourcc_s)

    writer = cv2.VideoWriter(dst, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise IOError(f"Could not open writer for destination: {dst}")

    fc = 0

    with tqdm(total=stopframe, unit='frame', desc='Writing video...') as pbar:
            while fc < stopframe:
                ret, frame = cap.read()
                if not ret:
                    # Source video ended before reaching stop_frame
                    break
                writer.write(frame)
                fc += 1
                pbar.update(1)

    print('\n'*2+'Video spliced!')
    cap.release()
    writer.release()

    return

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print('Correct script usage: python -m slice_avi [source path] [destination path] [stop frame] [fourcc codec (optional)]')
    elif len(sys.argv) == 4:
        splice_avi(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 5:
        splice_avi(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])