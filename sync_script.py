import numpy as np
import os
import pandas as pd
from pathlib import Path
import re

fps = 30

def make_ts_df(sess_path, sess_head, data_keys, ts_keys):

    raw_ts = {} # initiate dict to handle all csv's in

    for k in ts_keys: # loop through each key

        df = pd.read_csv(os.path.join(sess_path, f'{sess_head}_{k}_ts0.csv')) # read the csv and store as a DataFrame
        df = df[['Value', 'Timestamp.DateTime']] # condense DataFrame to the frame/byte value and timestamp
        df = df.rename(columns={'Timestamp.DateTime': 'Time'}) # rename column for ease of access
        df['Time'] = pd.to_datetime(df['Time'], utc=False, errors='coerce') # convert timestamps to DateTime objects
        
        if k == 'cam':
            start = min(df['Time'])

        raw_ts[k] = df # put DataFrame in the dict

    sync_ts = {k: {'Value': v['Value'], 'Time': (v['Time']-start).dt.total_seconds()}
               for k,v in raw_ts.items()}

    ref = sync_ts['cam'].sort_values('Time')

    for k,v in sync_ts.items():
        if k =='cam':
            continue

        merge = pd.merge_asof(v, ref, on='Time', direction='nearest', suffixes=['_raw','_sync'])
        merge['Diff'] = merge['Time_raw']-merge['Time_sync']
        mask = merge['Diff'] > (1/fps)
        merge.iloc[mask, 'Time_sync'] = None
        sync_ts['Time'] = merge['Time_sync']