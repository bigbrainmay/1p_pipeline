import numpy as np
import os
import pandas as pd
from pathlib import Path
import re

fps = 30

def make_ts_df(sess_path):
    
    # Handling behavioral camera .csv output
    frame_ts = pd.read_csv(os.path.join(sess_path, f'{subj}_{sess}_beh_ts.csv')) # read the .csv and save as a DataFrame
    frame_ts = frame_ts[['Value.Index', 'Timestamp.DateTime']] # condense to the frame index and timestamp columns
    frame_ts.columns = ['Value', 'Time'] # rename columns for ease of access
    frame_ts.insert(loc=0, column='Type', value='beh') # make a 'Type' column with values 'beh' for concatenation

    # Handling Bpod bytes .csv output
    bpod_ts = pd.read_csv(os.path.join(sess_path, f'{subj}_{sess}_bpod_ts.csv')) # read the .csv and save as a DataFrame
    bpod_ts = bpod_ts[['Value', 'Timestamp.DateTime']] # condense to the byte value and timestamp columns
    bpod_ts.columns = ['Value', 'Time'] # rename columns for ease of access
    bpod_ts.insert(loc=0, column='Type', value='bpod') # make a 'Type' column with values 'bpod' for concatenation

    # Handling miniscope .csv output
    scope_ts = pd.read_csv(os.path.join(sess_path, f'{subj}_{sess}_scope_ts.csv')) # read the .csv and save as a DataFrame
    scope_ts = scope_ts[['Value', 'Timestamp.DateTime']] # condense to the frame index and timestamp column
    scope_ts.columns = ['Value', 'Time'] # rename columns for ease of access
    scope_ts.insert(loc=0, column='Type', value='scope') # make a 'Type' column with values 'scope' for concatenation

    times = pd.concat([frame_ts, bpod_ts, scope_ts], axis=1) # concatenate behavioral camera, Bpod, and miniscope DataFrames
    start_ts = pd.to_datetime(time['Time'], utc=False, errors='coerce').min() # pull earliest timestamp
    times['Time'] = (times['Time']-start_ts).dt.total_seconds() # new column with time(s) elapsed since first timestamp

    ref_time = np.arange(0, max(ts['Time'].to_list()), 1/fps).to_list() # creates a reference/global timescale for alignment 

    


def frame_sync():



def main(sess_path, scope=True):
    