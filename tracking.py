import numpy as np
import pandas as pd

from config import px_to_cm, fps

def tracking_comps(data_synced):

    slp = data_synced['slp']
    if 'boris' in data_synced.keys():
        boris = data_synced['boris']
        starts = boris.loc[boris['Value']==0, 'Time'].to_numpy()
        stops = boris.loc[boris['Value']==4, 'Time'].to_numpy()
        unmask = np.logical_or.reduce([(slp['Time']>=i)&(slp['Time']<=f) for i,f in zip(starts,stops)])
        slp = slp[unmask].copy()
    fd = np.diff(slp['Frame'], prepend=0) # Number of frames between each slp entry

    slp['ear_rad'] = np.sqrt(
        ((slp['rear_x'] - slp['lear_x'])  **2) + 
        ((slp['rear_y'] - slp['lear_y'])  **2))/2.0 # half the length between ears(px)
    mear_x = (slp['lear_x']+slp['rear_x'])/2.0 # midway of ears x-axis pixel coordinate
    mear_y = (slp['lear_y']+slp['rear_y'])/2.0 # midway of ears y-axis pixel coordinate
    dx = np.diff(mear_x, prepend=0) * px_to_cm # delta x between each slp entry(cm)
    dy = np.diff(mear_y, prepend=0) * px_to_cm # delta y between each slp entry(cm) 
    dd = np.sqrt((dx**2)+(dy**2)) # distance traveled between each slp entry(cm)
    
    slp['hd'] = (np.degrees(np.arctan2(
        (slp['rear_y'] - mear_y), 
        (slp['rear_x'] - mear_x))) + 90) % 360 # head direction(degrees)
    slp['av'] = np.diff(slp['hd'], prepend=0) * fps # angular velocity(degrees/s)
    slp['v'] = np.diff(dd, prepend=0) * fps # linear velocity(cm/s)
    slp['acc'] = np.diff(slp['v'], prepend=0) # linear acceleration(cm/s^2)