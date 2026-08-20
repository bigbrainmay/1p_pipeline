import numpy as np
import random
from scipy.stats import percentileofscore

def shuffle_dist(gd, tests=None, state='Audio', shuffles=1000):
    
    from bpod_tools import pokes_in_range
    
    if tests=='all':
        tests=['compare','noncompare', 'compare_dr']
    elif not isinstance(tests, list):
        print('tests object must be a list.')
        return 1
    
    tests=[t.lower() for t in tests]
    
    results={
        'state': state, 
        'n_shuffles': shuffles, 
        }
    
    rng = np.random.default_rng()
    
    if 'compare' in tests:
        if state=='Audio':
            tdr=gd['atdr20']
        elif state=='Light':
            tdr=gd['ltdr20']
                
        offsets=rng.uniform(20, gd['trialStops'][-1], size=shuffles)
        
        shuffs1=[pokes_in_range(gd, state, duration=20, delay=t, trialType=1, for_shuffle=True) for t in offsets]
        shuffs2=[pokes_in_range(gd, state, duration=20, delay=t, trialType=2, for_shuffle=True) for t in offsets]
        
        if len(shuffs1)>len(shuffs2):
            shuffs1=shuffs1[:len(shuffs2)]
        elif len(shuffs1)<len(shuffs2):
            shuffs2=shuffs2[:len(shuffs1)]
        
        shuffs_timedr=[(t1['sumtp']-t2['sumtp'])/(t1['sumtp']+t2['sumtp']) for t1,t2 in zip(shuffs1,shuffs2)]   
        compare_percentile=percentileofscore(shuffs_timedr, tdr)
        results['compare_percentile'] = compare_percentile

    if 'noncompare' in tests:
       
        if state=='Audio':
            tdr=gd['t20d']
        elif state=='Light':
            tdr=gd['l20d']
                
        offsets=rng.uniform(20, gd['trialStops'][-1], size=shuffles)
                
        shuffs1=[pokes_in_range(gd, state, duration=20, delay=t, trialType=1, for_shuffle=True) for t in offsets]
        shuffs2=[pokes_in_range(gd, state, duration=20, delay=t-20, trialType=1, for_shuffle=True) for t in offsets]
        
        if len(shuffs1)>len(shuffs2):
            shuffs1=shuffs1[:len(shuffs2)]
        elif len(shuffs1)<len(shuffs2):
            shuffs2=shuffs2[:len(shuffs1)]
        
        shuffs_timedr=[t1['sumtp']-t2['sumtp'] for t1,t2 in zip(shuffs1,shuffs2)]   
        noncompare_percentile=percentileofscore(shuffs_timedr, tdr)
        results['noncompare_percentile'] = noncompare_percentile
    
    if 'compare_dr' in tests:
        
        if state=='Audio':
            tdr=gd['atdr20']
            samp=gd['tpi20']['totalPoke']+gd['npi20']['totalPoke']  
        elif state=='Light':
            tdr=gd['ltdr20']
            samp=gd['lpi20']['totalPoke']+gd['dpi20']['totalPoke']
        ht=int(len(samp)/2) # ht - half trial
        
        shuffle_drs=[]
        for i in range(shuffles):
            temp_pairs=random.sample(pairs,k=len(pairs))
            temp_csp=temp_pairs[:ht]
            temp_csm=temp_pairs[ht:]
            temp_ptd=sum([t[0]for t in temp_csp])-sum([t[1]for t in temp_csp])
            temp_mtd=sum([t[0] for t in temp_csm])-sum([t[1]for t in temp_csm])
            temp_dr=(temp_ptd-temp_mtd)/(temp_ptd+temp_mtd)
            shuffle_drs.append(temp_dr)
        
        compare_dr_percentile=percentileofscore(shuffle_drs, tdr)
        results['compare_dr_percentile']=compare_dr_percentile
    
    return results