import numpy as np
import os
import pandas as pd

from config import *
from utils import load_pkl, verify_len

def sync_data(sess_path, sess_head, ts_keys, data_keys):
    raw_ts = {} # initiate dict to handle all csv's in
    
    print(f'Timestamps used: {ts_keys}')
    print(f'Data being synced: {data_keys}'+'\n'*3)

    for k in ts_keys: # loop through each key

        df = pd.read_csv(os.path.join(sess_path, f'{sess_head}_{k}_ts0.csv')) # read the csv and store as a DataFrame
        df = df[['Value', 'Timestamp.DateTime']] # condense DataFrame to the frame/byte value and timestamp
        df = df.rename(columns={'Timestamp.DateTime': 'Time'}) # rename column for ease of access
        df['Time'] = pd.to_datetime(df['Time'], utc=False, errors='coerce') # convert timestamps to DateTime objects

        raw_ts[k] = df # put DataFrame in the dict

    start = raw_ts['cam']['Time'].min()

    sync_ts = {k: pd.DataFrame({'Value': v['Value'], 
                'Time': (v['Time']-start).dt.total_seconds()})
               for k,v in raw_ts.items()}

    ref = sync_ts['cam']
    ref.columns = ['Frame', 'Time_sync']

    for k,v in sync_ts.items():
        if k =='cam':
            continue

        merge = pd.merge_asof(v, ref, left_on='Time', right_on='Time_sync', direction='nearest')
        merge['Diff'] = np.abs(merge['Time']-merge['Time_sync'])
        mask = merge['Diff'] > (1/fps)
        merge.loc[mask, ['Frame','Time_sync']] = np.nan
        v['Time'] = merge['Time_sync']
        v.insert(loc=0, column='Frame', value=merge['Frame'])
        sync_ts[k]=v

    print('Timestamps synced to behavior camera!'+'\n'*3+'Loading data...'+'\n'*3)

    data_dfs = {}

    if 'boris' in data_keys:
        df = pd.read_csv(os.path.join(sess_path, f'{sess_head}_boris1.csv'))
        df = df[['Behavior','Behavior type','Image index']]
        df.columns = ['Event','Type','Frame']
        df.insert(loc=0, column='Value', value=0), df['Value']=df['Value'].astype(object)
        df.loc[df['Type']=='STOP', 'Value'] = 4
        df.loc[~df['Type'].isin(['START','STOP']), 'Value'] = df.loc[~df['Type'].isin(['START','STOP']), 'Event']
        data_dfs['boris'] = df
        print('BORIS data loaded!'+'\n')
    
    if 'bpod' in data_keys:
        dct = load_pkl(os.path.join(sess_path,f'{sess_head}_bpod.pkl'))
        type1s = np.array(dct['Types']==1)
        df = pd.DataFrame({k:v for k,v in dct['concatTable'] if k in ['Name','Total']})
        df = df.rename(columns={'Total':'HW_time'})
        df.insert(loc=0,column='Value',value=0)
        df.loc[df['Name']=='ITI','Value'] = 2
        if 'Audio' in df['Name'].values:
            df.loc[df['Name']=='Audio','Value'] = np.where(type1s,3,4)
        if 'Light' in df['Name'].values:
            df.loc[df['Name']=='Light','Value'] = np.where(type1s,5,6)
        df['Value'] = df['Value'].astype('int64')
        data_dfs['bpod'] = df
        print('Bpod data loaded!'+'\n')
    
    if 'slp' in data_keys:
        df = pd.read_csv(os.path.join(sess_path,f'{sess_head}_SLP.csv'))
        df = df.drop(columns=['track','instance.score','rcap.score','lcap.score','rear.score','lear.score','butt.score'])
        df = df.rename(columns={'frame_idx':'Frame'})
        df.columns = [c.replace('.','_') for c in df.columns.to_list()]
        data_dfs['slp'] = df
        print('SLP data loaded!'+'\n')

    if 'cells' in data_keys:
        data_dfs['cells'] = pd.read_csv(os.path.join(sess_path,f'{sess_head}_cells.csv'))
        print('Cell data loaded!'+'\n')

    print('\n'+'All data loaded. Syncing data...'+'\n'*3)
    data_synced = {}

    if 'boris' in data_keys:
        df = data_dfs['boris']
        ts = sync_ts['cam']
        df = df.merge(ts[['Frame','Time']],on='Frame',how='left')
        data_synced['boris'] = df
        print('BORIS data synced!'+'\n')

    if 'bpod' in data_keys:
        df = data_dfs['bpod']
        ts = sync_ts['bpod']
        df = df.merge(ts[['Value','Time']],on='Value',how='left')
        data_synced['bpod'] = df
        print('Bpod data synced!'+'\n')

    if 'slp' in data_keys:
        df = data_dfs['slp']
        ts = sync_ts['cam']
        df = df.merge(ts[['Frame','Time']],on='Frame',how='left')
        data_synced['slp'] = df
        print('SLP data synced!'+'\n')

    if 'cells' in data_keys:
        df = data_dfs['cells']
        ts = sync_ts['scope']
        if verify_len(df,ts):
            df.insert(loc=0,column='Time',value=ts['Time'])
            data_synced['cells'] = df
            print('Cell data synced!'+'\n')
        else:
            print('The number of timepoints for cells and miniscope frames received by Bonsai do not match. Troubleshooting required.')
            return

    return data_dfs