import os
import tqdm

CHUNK_SIZE = 1024*1024*4

def load_large_binary(filepath):
    try:
        with open(filepath, 'rb') as f:
            print(f'Loading binary file {filepath}...')
            data = f.read()
            return data
    except Exception as e:
        print(f'{e}')