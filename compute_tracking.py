import numpy as np
import pandas as pd

from config import *
from utils import arr_caps, diff_step

def compute_tracking(data_synced):

    slp = data_synced['slp']
    if 'boris' in data_synced.keys():
        boris = data_synced['boris']
        starts = boris.loc[boris['Value']==0, 'Time'].to_numpy() # 0 is code for trial starts
        stops = boris.loc[boris['Value']==4, 'Time'].to_numpy() # 4 is code for trial stops
        unmask = np.logical_or.reduce([(slp['Time']>=i)&(slp['Time']<=f) for i,f in zip(starts,stops)])
        slp = slp[unmask].copy()
    td_step = diff_step(slp['Time'], slp_step, cap=True) # Time elapsed between each slp entry

    mear_x = np.array([np.nanmean([l,r]) for l,r in slp[['lear_x', 'rear_x']].to_numpy()]) # midway of ears x-axis pixel coordinate 
    mear_y = np.array([np.nanmean([l,r]) for l,r in slp[['lear_y', 'rear_y']].to_numpy()]) # midway of ears y-axis pixel coordinate
    dx = diff_step(mear_x, slp_step, arr_caps=True) * px_to_cm # delta x(cm)
    dy = diff_step(mear_y, slp_step, arr_caps=True) * px_to_cm # delta y(cm) 
    dd = np.sqrt((dx**2)+(dy**2)) # distance traveled(cm)
    
    slp['hd'] = (np.degrees(np.arctan2(
        (slp['rear_y'] - mear_y), 
        (slp['rear_x'] - mear_x))) + 90) % 360 # head direction(degrees)
    hd = slp['hd'].to_numpy()
    slp['av'] = diff_step(slp['hd'], slp_step, cap=True) * td_step # angular velocity(degrees/s)
    slp['v'] = diff_step(dd, slp_step, cap=True) * td_step # linear velocity(cm/s)
    slp['acc'] = diff_step(slp['v'], slp_step, cap=True) # linear acceleration(cm/s^2)

    slp.loc[(td_step>td_cap), ['av', 'v', 'acc']] = np.nan
    slp.loc[(slp['v']>v_cap), ['v', 'acc']] = np.nan
    slp.loc[(slp['acc']>acc_cap), 'acc'] = np.nan

    cw_adj = [(i<90) and (f>270) for i,f in zip(hd[:-slp_step], hd[slp_step:])]
    cw_adj = arr_caps(len(slp), cw_adj)
    ccw_adj = [(i>270) and (f<90) for i,f in zip(hd[:-slp_step], hd[slp_step:])]
    ccw_adj = arr_caps(len(slp), ccw_adj)
    slp.loc[cw_adj, 'av'] = 360 - slp.loc[cw_adj, 'av']
    slp.loc[ccw_adj, 'av'] = 360 + slp.loc[ccw_adj, 'av']

    return slp