import numpy as np
import os
import sys

def load_npy_file(filepath):
    if os.path.exists(filepath):
        data = np.load(filepath)
        return data
    else:
        raise FileNotFoundError(f'File does not exist: {filepath}')
    
if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise ValueError('Please provide .npy file to view as argument.')
    fp = sys.argv[1]
    file = load_npy_file(fp)
    shape_or_print = input('For shape type "s", for full print type "p": ')
    if shape_or_print == 's':
        print(file.shape)
    elif shape_or_print == 'p':
        print(file)
    else:
        raise ValueError('Invalid input. Please enter "s" or "p".')