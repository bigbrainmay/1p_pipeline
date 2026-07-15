import os
import re

date_format = "\\d{2}_\\d{2}_\\d{4}"
day_format = '\\d{8}'


DATA_INDS = {
    'rat_name_conv': r"[A-Za-z]{2}\d{2}",
    'sess_name_conv': f'^{day_format}(_.)?$',
    'JB_experiments': ['CCH_7day', 'CCH_9day', 'CCH_40trial_10day', 'CCH_50trial_10day', 
                    'CCH_Mirror', 'CH_2day','CH_3day', 'CH_6day', 'Curtain_1well_AL'],
    'JL_experiments': ['SPC_5day_1', 'SPC_5day_2'],
    'video_ts': r"\d{4}-\d{2}-\d{2}[A-Za-z]\d{2}_\d{2}_\d{2}.csv",
    'boris': ['boris1', 'boris2', 'boris3'],
    'pos': ['_SLP_', '_3pt_', 'slp'],
    'trial_df': ['trial_metrics', 'trial_df'],
    'ephys': [r'test_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}'],
    'ephys_rec_path': os.path.join('Record Node 101', 'experiment1', 'recording1', 'continuous', 
                        'Acquisition_Board-100.Rhythm Data'),
    'ephys_ttl_path': os.path.join('Record Node 101', 'experiment1', 'recording1', 'events', 
                                   'Acquisition_Board-100.Rhythm Data', 'TTL', 'timestamps.npy'),
    'ks': ['kilosort(\\d)?', 'ksOut(\\d)?'],
    'prot': ['Habituation', 'PreconditioningJL', 'ConditioningJL', 'Test1JL', 'Test2JL'],
    'slp_models': ['nocapchz', 'capchz', 'chz_fe_labels', 'box_fe_labels', 'capTM_labels'],
    'OE_refresh': 30000,
} 

IND_COMP = {
    'JB_exp': re.compile('|'.join(DATA_INDS['JB_experiments'])),
    'JL_exp': re.compile('|'.join(DATA_INDS['JL_experiments'])),
    'boris': re.compile('|'.join(DATA_INDS['boris'])),
    'pos': re.compile('|'.join(DATA_INDS['pos'])),
    'trial_df': re.compile('|'.join(DATA_INDS['trial_df'])),
    'ephys': re.compile('|'.join(DATA_INDS['ephys'])),
    'ks': re.compile('|'.join(DATA_INDS['ks'])),
    'prot': re.compile('|'.join(DATA_INDS['prot'])),
    'slp_models': re.compile('|'.join(DATA_INDS['slp_models']))
}