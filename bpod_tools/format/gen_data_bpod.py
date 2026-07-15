import numpy as np
import os
from pathlib import Path
import re

def gen_data_bpod(sess_path, gd=None, save=False):
    
    sess_path = r"{}".format(sess_path)
    
    import matlab.engine
    import pickle
    
    from bpod_tools import pokes_in_range, shuffle_dist
    
    if not gd:
        funcpath = Path(__file__)
        gdbp = fr'{funcpath.parent}'
        print(gdbp)
        mtpp = os.path.join(fr'{funcpath.parents[2]}', 'lab_utils', 'transforms')
        print('\n' + 'Starting MATLAB engine...' + '\n')
        engine = matlab.engine.start_matlab()
        print('MATLAB engine started!')
        engine.addpath(gdbp, nargout=0)
        engine.addpath(mtpp, nargout=0)
        gd = engine.gen_data_bpod(sess_path, True)
        
        gd['trialStarts'] = np.array(gd['trialStarts'])
        gd['trialStarts'] = [float(x[0]) for x in gd['trialStarts']]
        gd['trialStops'] = np.array(gd['trialStops'])
        gd['trialStops'] = [float(x[0]) for x in gd['trialStops']]
        if 'Types' in gd:
            gd['Types'] = np.array(gd['Types'])[0]
            gd['Types'] = [int(x) for x in gd['Types']]
        
        if 'probeTypes' in gd:
            gd['probeTypes'] = np.array(gd['probeTypes'])[0]
            gd['probeTypes'] = [int(x) for x in gd['probeTypes']]
            gd['probeStarts'] = np.array(gd['probeStarts'])
            gd['probeStarts'] = [float(x[0]) for x in gd['probeStarts']]
            gd['probeStops'] = np.array(gd['probeStops'])
            gd['probeStops'] = [float(x[0]) for x in gd['probeStops']]
    
    desc = gd['desc']
    prot = desc['protocol']
    subj = desc['subject']
    sess = desc['session']
    
    ct = gd['concatTable'] # ct - concat table
    en = set([e['Name'] for e in ct if not re.match(r'^\d+$', e['Name'])]) # en - event names

    if 'Light' in en:
        if 'Types' in gd:
            gd['lpi'] = pokes_in_range(gd, 'Light', trialType=1) # lpi - light pokes initial
            gd['lpf'] = pokes_in_range(gd, 'Light', onset=False, trialType=1) # lpf - light pokes final
            gd['lpi20'] = pokes_in_range(gd, 'Light', duration=20, trialType=1)
            gd['dpi'] = pokes_in_range(gd, 'Light', trialType=2) # dpi - dim pokes initial
            gd['dpf'] = pokes_in_range(gd, 'Light', onset=False, trialType=2) # dpf - dim pokes final
            gd['dpi20'] = pokes_in_range(gd, 'Light', duration=20, trialType=2)
            gd['lpp'] = pokes_in_range(gd, 'Light', delay = -10, trialType=1) # lpp - light pokes prior
            gd['lpp20'] = pokes_in_range(gd, 'Light', duration=20, delay=-20, trialType=1)
            gd['dpp'] = pokes_in_range(gd, 'Light', delay = -10, trialType=2) # dpp - dim pokes prior
            gd['dpp20'] = pokes_in_range(gd, 'Light', duration=20, delay=-20, trialType=2)
            gd['ld'] = gd['lpi']['sumtp']-gd['lpp']['sumtp']
            gd['dd'] = gd['dpi']['sumtp']-gd['dpp']['sumtp']
            gd['l20d'] = gd['lpi20']['sumtp']-gd['lpp20']['sumtp']
            gd['d20d'] = gd['dpi20']['sumtp']-gd['dpp20']['sumtp']
            if gd['lpi']['sumtp'] == 0 and gd['dpi']['sumtp'] == 0:
                gd['ltdr'] = 0
                gd['lcdr'] = 0
                gd['lbdr'] = 0
            else:
                gd['ltdr'] = (gd['lpi']['sumtp']-gd['dpi']['sumtp'])/(gd['lpi']['sumtp']+gd['dpi']['sumtp']) # ltdr - light time disc. ratio
                gd['lcdr'] = (sum(gd['lpi']['count'])-sum(gd['dpi']['count']))/(sum(gd['lpi']['count'])+sum(gd['dpi']['count'])) # light lcdr - count disc. ratio
                gd['lbdr'] = (gd['lpi']['responses']-gd['dpi']['responses'])/(gd['lpi']['responses']+gd['dpi']['responses']) # lbdr - light binary disc. ratio
            if gd['lpi20']['sumtp'] == 0 and gd['dpi20']['sumtp'] == 0:
                gd['ltdr20'] = 0
                gd['lcdr20'] = 0
                gd['lbdr20'] = 0
            else:
                gd['ltdr20'] = (gd['lpi20']['sumtp']-gd['dpi20']['sumtp'])/(gd['lpi20']['sumtp']+gd['dpi20']['sumtp'])
                gd['lcdr20'] = (sum(gd['lpi20']['count'])-sum(gd['dpi20']['count']))/(sum(gd['lpi20']['count'])+sum(gd['dpi20']['count']))
                gd['lbdr20'] = (gd['lpi20']['responses']-gd['dpi20']['responses'])/(gd['lpi20']['responses']+gd['dpi20']['responses'])
            
            if len(gd['lpi']['count'])<=len(gd['dpi']['count']):
                lnp=len(gd['lpi']['count'])
            else:
                lnp=len(gd['dpi']['count'])
            gd['ltdrt'] = [0 if gd['lpi']['totalPoke'][i]==0 and gd['dpi']['totalPoke'][i]==0 else (gd['lpi']['totalPoke'][i]-gd['dpi']['totalPoke'][i])/(gd['lpi']['totalPoke'][i]+gd['dpi']['totalPoke'][i]) for i in range(lnp)]
            gd['lcdrt'] = [0 if gd['lpi']['count'][i]==0 and gd['dpi']['count'][i]==0 else (gd['lpi']['count'][i]-gd['dpi']['count'][i])/(gd['lpi']['count'][i]+gd['dpi']['count'][i]) for i in range(lnp)]
            gd['lbdrt'] = [(gd['lpi']['count'][i]>0)-(gd['dpi']['count'][i]>0) for i in range(lnp)]
            gd['ltdr20t'] = [0 if gd['lpi20']['totalPoke'][i]==0 and gd['dpi20']['totalPoke'][i]==0 else  (gd['lpi20']['totalPoke'][i]-gd['dpi20']['totalPoke'][i])/(gd['lpi20']['totalPoke'][i]+gd['dpi20']['totalPoke'][i]) for i in range(lnp)]
            gd['lcdr20t'] = [0 if gd['lpi20']['count'][i]==0 and gd['dpi20']['count'][i]==0 else  (gd['lpi20']['count'][i]-gd['dpi20']['count'][i])/(gd['lpi20']['count'][i]+gd['dpi20']['count'][i]) for i in range(lnp)]
            gd['lbdr20t'] = [(gd['lpi20']['count'][i]>0)-(gd['dpi20']['count'][i]>0) for i in range(lnp)]
            
            gd['light_shuffles'] = shuffle_dist(gd, tests='all', state='Light')
            
        else:
            gd['lpi'] = pokes_in_range(gd, 'Light') # lpi - light pokes initial
            gd['lpf'] = pokes_in_range(gd, 'Light', onset=False) # lpf - light pokes final
            gd['lpp'] = pokes_in_range(gd, 'Light', delay = -10) # lpp - light pokes prior
            
    if 'Audio' in en:
        gd['tpi'] = pokes_in_range(gd, 'Audio', trialType=1) # tpi - tone pokes initial
        gd['tpf'] = pokes_in_range(gd, 'Audio', onset=False, trialType=1) # tpf - tone pokes final
        gd['tpi20'] = pokes_in_range(gd, 'Audio', duration=20, trialType=1)
        gd['npi'] = pokes_in_range(gd, 'Audio', trialType=2) # npi - noise pokes initial
        gd['npf'] = pokes_in_range(gd, 'Audio', onset=False, trialType=2) # npf - noise pokes final
        gd['npi20'] = pokes_in_range(gd, 'Audio', duration=20, trialType=2)
        gd['tpp'] = pokes_in_range(gd, 'Audio', delay=-10, trialType = 1) # tpp - tone pokes prior
        gd['tpp20'] = pokes_in_range(gd, 'Audio', duration=20, delay=-20, trialType=1)
        gd['npp'] = pokes_in_range(gd, 'Audio', delay=-10, trialType=2) # npp - noise pokes prior
        gd['npp20'] = pokes_in_range(gd, 'Audio', duration=20, delay=-20, trialType=2)
        gd['td'] = gd['tpi']['sumtp']-gd['tpp']['sumtp']
        gd['dd'] = gd['npi']['sumtp']-gd['npp']['sumtp']
        gd['t20d'] = gd['tpi20']['sumtp']-gd['tpp20']['sumtp']
        gd['n20d'] = gd['npi20']['sumtp']-gd['npp20']['sumtp']
        if gd['tpi']['sumtp'] == 0 and gd['npi']['sumtp'] == 0:
            gd['atdr'] = 0
            gd['acdr'] = 0
            gd['abdr'] = 0
        else:
            gd['atdr'] = (gd['tpi']['sumtp']-gd['npi']['sumtp'])/(gd['tpi']['sumtp']+gd['npi']['sumtp']) # atdr - audio time disc. ratio
            gd['acdr'] = (sum(gd['tpi']['count'])-sum(gd['npi']['count']))/(sum(gd['tpi']['count'])+sum(gd['npi']['count'])) # acdr - audio count disc. ratio
            gd['abdr'] = (gd['tpi']['responses']-gd['npi']['responses'])/(gd['tpi']['responses']+gd['npi']['responses']) # abdr - audio binary disc. ratio
        if gd['tpi20']['sumtp'] == 0 and gd['npi20']['sumtp'] == 0:
            gd['atdr20'] = 0
            gd['acdr20'] = 0
            gd['abdr20'] = 0
        else:
            gd['atdr20'] = (gd['tpi20']['sumtp']-gd['npi20']['sumtp'])/(gd['tpi20']['sumtp']+gd['npi20']['sumtp'])
            gd['acdr20'] = (sum(gd['tpi20']['count'])-sum(gd['npi20']['count']))/(sum(gd['tpi20']['count'])+sum(gd['npi20']['count']))
            gd['abdr20'] = (gd['tpi20']['responses']-gd['npi20']['responses'])/(gd['tpi20']['responses']+gd['npi20']['responses'])
        
        if len(gd['tpi']['count'])<=len(gd['npi']['count']):
            anp=len(gd['tpi']['count'])
        else:
            anp=len(gd['npi']['count'])
        gd['atdrt'] = [0 if gd['tpi']['totalPoke'][i]==0 and gd['npi']['totalPoke'][i]==0 else (gd['tpi']['totalPoke'][i]-gd['npi']['totalPoke'][i])/(gd['tpi']['totalPoke'][i]+gd['npi']['totalPoke'][i]) for i in range(anp)]
        gd['acdrt'] = [0 if gd['tpi']['count'][i]==0 and gd['npi']['count'][i]==0 else (gd['tpi']['count'][i]-gd['npi']['count'][i])/(gd['tpi']['count'][i]+gd['npi']['count'][i]) for i in range(anp)]
        gd['abdrt'] = [(gd['tpi']['count'][i]>0)-(gd['npi']['count'][i]>0) for i in range(anp)]
        gd['atdr20t'] = [0 if gd['tpi20']['totalPoke'][i]==0 and gd['npi20']['totalPoke'][i]==0 else  (gd['tpi20']['totalPoke'][i]-gd['npi20']['totalPoke'][i])/(gd['tpi20']['totalPoke'][i]+gd['npi20']['totalPoke'][i]) for i in range(anp)]
        gd['acdr20t'] = [0 if gd['tpi20']['count'][i]==0 and gd['npi20']['count'][i]==0 else (gd['tpi20']['count'][i]-gd['npi20']['count'][i])/(gd['tpi20']['count'][i]+gd['npi20']['count'][i]) for i in range(anp)]
        gd['abdr20t'] = [(gd['tpi20']['count'][i]>0)-(gd['npi20']['count'][i]>0) for i in range(anp)]
        
        gd['audio_shuffles'] = shuffle_dist(gd, tests='all')
        
    if save:
        with open(os.path.join(sess_path, f'{subj}_{sess}_gen_data_bpod.pkl'), 'wb') as f:
             pickle.dump(gd, f)
             
    gen_data = gd
    return gen_data

if __name__ == "__main__":
    out = gen_data_bpod(sp,gd,sv)
    
# Difference of Pre-post for CS+ and pre-post for CS- / Their sum
# Do that with shuffled stimuli groupings, find percentile of that + basic t-test