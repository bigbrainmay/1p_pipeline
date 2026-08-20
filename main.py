import os
from pathlib import Path
import sys

from config import wd
from utils import *

def sess_to_local(server_path):

    sess = Path(server_path).name
    sess_path = os.path.join(wd, sess)

    sess_copy_ver = folder_copy(server_path, sess_path)
    if not sess_copy_ver:
        print(f'Pipeline halted. Unable to move session folder to local directory.')
        return

    return sess_path

        