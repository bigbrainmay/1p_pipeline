import numpy as np

env_w = 30.48 # 1ft. in cm; to be measured later

# To be replaced with soft-coded reading of manually selected ROI coordinates:
corners = [(0,400),(400,400)] # random hard-coded pixel coords. for now

env_px = np.sqrt(
    ((corners[1][0]-corners[0][0])**2) + 
    ((corners[1][1]-corners[0][1])**2)) # Computes pixel length between adjacent environment edges

px_to_cm = env_px/env_w # Scalar to convert pixel lengths to cm in environment

fps = 30