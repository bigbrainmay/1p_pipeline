import os
import sys
import tqdm

chunk_size = 1024*1024*4

def verify_binary_concat(src_files, dst_path):
    print('Verifying concatenation...')
    try:
        with open(dst_path, "rb") as out_f:
            for i, f in enumerate(src_files):
                with open(f, "rb") as sess_f:
                    while True:
                        sess_chunk = sess_f.read(chunk_size)
                        out_chunk = out_f.read(len(sess_chunk))
                        if not sess_chunk:
                            print(f'Session {i+1} of {len(src_files)} data match verified!')
                            break
                        if sess_chunk != out_chunk:
                            print(f"Mismatch detected in: {f}")
                            return False
        return True
    except Exception as e:
        print(f"Error during verification: {e}")
        return False

def binary_file(src_files, dst_file):

    with open(dst_file, 'wb') as dst:
        for i, f in enumerate(src_files):

            fsize = os.path.getsize(f)

            print(f'Concatenating file {i+1} of {len(src_files)}')
            try:
                   with open(f, 'rb') as src: 
                        
                        with tqdm(
                            total=fsize, unit='B', unit_scale=True, unit_divisor=1024,
                            desc=f'{i+1} of {len(src_files)}', ncols=80, leave=False
                        ) as pbar:
                            
                            while chunk := src.read(chunk_size):
                                dst.write(chunk)
                                pbar.update(len(chunk))

            except FileNotFoundError:
                print(f'File does not exist: {f}')
        
    print(f'Finished concatenation to path: {dst_file}')

    if verify_binary_concat(src_files, dst_file):
        print('concatenation verified.')
    else:
        print('concatenation not verified. See above for issue.')

    
    if __name__ == '__main__':
        if len(sys.argv) < 2:
            raise ValueError('Please provide file paths for binary file concatenation.')        
        srcs = sys.argv[1:-1]
        dst = sys.argv[-1]
        binary_file(srcs, dst)