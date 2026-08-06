import numpy as np



# To be replaced with soft-coded reading of manually selected ROI coordinates:


fps=30


# SLP config
# -----------

slp_step = 2
td_cap = 1.5 # Maximum time diff. of SLP frames before omitting av, v, acc data
corners = [(0,400),(400,400)] # random hard-coded pixel coords. for now
env_px = np.sqrt( # Computes pixel length between adjacent environment edges
    ((corners[1][0]-corners[0][0])**2) + 
    ((corners[1][1]-corners[0][1])**2)) 
env_w = 30.48 # 1ft. in cm; to be measured later
px_to_cm = env_px/env_w # Scalar to convert pixel lengths to cm in environment
av_cap = 120 # Highest valid angular velocity
v_cap = 50 # Highest valid velocity calculation
acc_cap = 100 # Highest valid acceleration calculation