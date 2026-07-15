import joblib
import pickle
import sys


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
    
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError('Please provide .pkl file path as an argument')
    
    fp = sys.argv[1]
    obj = load_pkl(fp)
    print('\n'*2)
    print(obj)