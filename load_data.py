import numpy as np
import os
import pandas as pd

from utils import load_pkl, verify_len

def make_data_dfs(sess_path, sess_head, data_keys):

    data_dfs = {}

    if 'boris' in data_keys:
        df = pd.read_csv(os.path.join(sess_path, f'{sess_head}_boris1.csv'))
        df = df[['Behavior','Behavior type','Image index']]
        df.columns = ['Event','Type','Frame']
        df.insert(loc=0, column='Value', value=0), df['Value']=df['Value'].astype(object)
        df.loc[df['Type']=='STOP', 'Value'] = 4
        df.loc[~df['Type'].isin(['START','STOP']), 'Value'] = df.loc[~df['Type'].isin(['START','STOP']), 'Event']
        data_dfs['boris'] = df
    
    if 'bpod' in data_keys:
        dct = load_pkl(os.path.join(sess_path,f'{sess_head}_bpod_data.pkl'))
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
    
    if 'slp' in data_keys:
        df = pd.read_csv(os.path.join(sess_path,f'{sess_head}_SLP.csv'))
        df = df.drop(columns=['track','instance.score','rcap.score','lcap.score','rear.score','lear.score','butt.score'])
        df = df.rename(columns={'frame_idx':'Value'})
        df.columns = [c.replace('.','_') for c in df.columns.to_list()]
        data_dfs['slp'] = df

    if 'cells' in data_keys:
        data_dfs['cells'] = pd.read_csv(os.path.join(sess_path,f'{sess_head}_cells.csv'))

def sync_data_dfs(data_keys,data_dfs,ts_dfs):

    data_synced = {}

    if 'boris' in data_keys:
        df = data_dfs['boris']
        ts = ts_dfs['cam']
        df = df.merge(ts[['Value','Time']],on='Value',how='left')
        data_synced['boris'] = df

    if 'bpod' in data_keys:
        df = data_dfs['bpod']
        ts = ts_dfs['bpod']
        df = df.merge(ts[['Value','Time']],on='Value',how='left')
        data_synced['bpod'] = df

    if 'slp' in data_keys:
        df = data_dfs['slp']
        ts = ts_dfs['cam']
        df = df.merge(ts[['Value','Time']],on='Value',how='left')
        data_synced['slp'] = df

    if 'cells' in data_keys:
        df = data_dfs['cells']
        ts = ts_dfs['scope']
        if verify_len(df,ts):
            df.insert(loc=0,column='Time',value=ts['Time'])
            data_synced['cells'] = df
        else:
            print('The number of timepoints for cells and miniscope frames received by Bonsai do not match. Troubleshooting required.')

    