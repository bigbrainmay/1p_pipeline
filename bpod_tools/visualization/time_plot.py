def time_plot(prot_gd_dict):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    pgd = prot_gd_dict
    keys = list(pgd.keys())
    
    root = r"Z:\Data\Julia\Jayce\data"
    
    exp = pgd[keys[0]]['desc']['experiment']
    prot = pgd[keys[0]]['desc']['protocol']
    subj = pgd[keys[0]]['desc']['subject']
    
    ### FILL OUT LAST FIELD YOURSELF ###
    tempdir = os.path.join(root,exp,'figures',f'{subj}_{prot}_')
        
    figsize = (6,4)
    tbarx = [x+0.375 for x in list(range(len(keys)))]
    nbarx = [x+0.625 for x in list(range(len(keys)))]
    barw = 0.25
    templen = sum([pgd[k]['nTrials'] for k in keys])
    tempx = [i for i in range(int(templen)+1)]
    
    xmax = templen+2
    if templen < 75:
        xticks = list(range(0,int(templen)+5,5))
    else:
        xticks = list(range(0,int(templen)+10,10))
    bar_lab_x = [x+0.5 for x in list(range(len(keys)))]
    bar_labs = [x+1 for x in list(range(len(keys)))]
    
    start = 1
    day_start = []
    if 'Preconditioning' in prot or 'Test' in prot:
        fd = {
            'tpic': [],
            'npic': [],
            'types': [],
            'pptp': [],
            'ppnp': [],
            'tonex': [],
            'noisex': []
            }
        if 'Preconditioning' in prot:
            fd['lpic'] = []
        for k,v in pgd.items():
            trials = list(range(int(v['nTrials'])))
            ttpi = [c for c in v['tpi']['count'] if c > 0]
            tnpi = [c for c in v['npi']['count'] if c > 0]
            ttx = [i+start for i,t in enumerate(v['Types']) if t == 1]
            ttx = [i for n,i in enumerate(ttx) if v['tpi']['count'][n] > 0]
            tnx = [i+start for i,t in enumerate(v['Types']) if t == 2]
            tnx = [i for n,i in enumerate(tnx) if v['npi']['count'][n] > 0]
            
            fd['pptp'].append(v['pptp'])
            fd['ppnp'].append(v['ppnp'])
            fd['tpic'].extend(ttpi)
            fd['npic'].extend(tnpi)
            fd['tonex'].extend(ttx)
            fd['noisex'].extend(tnx)
            
            if 'Test' in prot:
                ttpi20 = [c for c in v['tpi20']['count'] if c > 0]
                tnpi20 = [c for c in v['npi20']['count'] if c > 0]
                tt20x = [i+start for i,t in enumerate(v['Types']) if t == 1]
                tt20x = [i for n,i in enumerate(tt20x) if v['tpi20']['count'][n] > 0]
                tn20x = [i+start for i,t in enumerate(v['Types']) if t == 2]
                tn20x = [i for n,i in enumerate(tn20x) if v['npi20']['count'][n] > 0]
                
                fd20['pptp20'].append(v['pptp20'])
                fd20['ppnp20'].append(v['ppnp20'])
                fd20['tpi20c'].extend(ttpi20)
                fd20['npi20c'].extend(tnpi20)
                fd20['tone20x'].extend(tt20x)
                fd20['noise20x'].extend(tn20x)
                
            if 'Preconditioning' in prot:
                tlpi = [c for c in v['lpi']['count'] if c > 0]
                fd['lpic'].extend(tlpi)
        
            start += v['nTrials']
            day_start.append(start)
        
        pcymax = int(max(fd['tpic']+fd['npic']))
        pcymax = int(max(fd['tpic']+fd['npic']))
        if pcymax < 20:
            pcyticks = list(range(pcymax+1))
            pcymax += 0.2
        elif pcymax < 30: 
            pcyticks = list(range(0,pcymax+1,2))
            pcymax += 0.4
        else:
            pcyticks = list(range(0,pcymax+1,4))
            pcymax += 0.8
        pcplt = plt.figure(figsize = figsize)
        plt.scatter(fd['tonex'], fd['tpic'], color='pink')
        plt.scatter(fd['noisex'], fd['npic'], color='black')
        plt.ylim(0,pcymax+0.2)
        plt.yticks(pcyticks)
        plt.xlim(0,xmax)
        plt.xticks(xticks)
        plt.title('Pokes during audio stimuli (10s)')
        plt.ylabel('Count')
        plt.xlabel('Trial')
        ax = plt.gca()
        ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        if len(day_start) > 1:
            for t in day_start[:-1]:
                ax.axvline(x=t, color = 'gray', linestyle='--', linewidth=1, alpha=0.7)
        plt.show()
        # pcplt.savefig(tempdir + 'trial_pokes_10s.png')
        
        pptpymax = np.abs(round(max(fd['pptp']+fd['ppnp'], key=abs), 2))
        pptpymin = np.abs(round(min(fd['pptp']+fd['ppnp'], key=abs), 2))
        if pptpymin < 0.005:
            pptpyticks = list(range(int((pptpymax+0.005)*-100), int((pptpymax+0.02)*100), 2))
            pptpymax += 0.005
        else:
            pptpyticks = list(range(int((pptpymax+0.05)*-100), int((pptpymax+0.06)*100), 2))
            pptpymax += 0.01
        pptpplt = plt.figure(figsize=figsize)
        plt.bar(tbarx, fd['pptp'], barw, color='pink')
        plt.bar(nbarx, fd['ppnp'], barw, color='black')
        plt.ylim(-pptpymax,pptpymax)
        plt.yticks([i*0.01 for i in pptpyticks])
        plt.xlim(0, len(keys))
        plt.xticks(bar_lab_x, bar_labs)
        plt.title('Pre/post stimulus port time difference')
        plt.ylabel('Percent difference (post - pre)')
        ax = plt.gca()
        ax.spines['bottom'].set_position(('data', pptpyticks[0]*0.01))
        ax.set_xticks(bar_lab_x)
        ax.set_xticklabels(bar_labs)
        ax.set_xlabel('Day')
        plt.axhline(0, color = 'black', linewidth=1)
        plt.show()
        pptpplt.savefig(tempdir + 'pre_post_time.png')
        
    elif 'Conditioning' in prot:
        fd = {
            'ccpc': [],
            'lpic': [],
            'ppsp': [],
            'ccx': [],
            'lx': []
        }
        for k,v in pgd.items():
            trials = list(range(int(v['nTrials'])))
            tccp = [c for c in v['ccp']['count'] if c > 0]
            tccx = [n+start for n,c in enumerate(v['ccp']['count']) if c > 0]
            tlpi = [c for c in v['lpi']['count'] if c > 0]
            tlx = [n+start for n,c in enumerate(v['lpi']['count']) if c > 0]
            fd['ccpc'].extend(tccp)
            fd['lpic'].extend(tlpi)
            fd['ppsp'].append(v['ppsp'])
            fd['ccx'].extend(tccx)
            fd['lx'].extend(tlx)
            
            start += int(v['nTrials'])
            day_start.append(start)
            
        ccymax = int(max(fd['ccpc']))
        if ccymax < 20:
            ccyticks = list(range(ccymax+1))
            ccymax += 0.2
        elif ccymax < 30: 
            ccyticks = list(range(0,ccymax+1,2))
            ccymax += 0.4
        else:
            ccyticks = list(range(0,ccymax+1,4))
            ccymax += 0.8
        cc2plt = plt.figure(figsize = figsize)
        plt.scatter(fd['ccx'],fd['ccpc'],color='pink')
        plt.ylim(0,ccymax)
        plt.yticks(ccyticks)
        plt.xlim(0,xmax)
        plt.xticks(xticks)
        plt.title('Pokes following control clicks (10s)')
        plt.ylabel('Count')
        plt.xlabel('Trial')
        ax = plt.gca()
        ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        if len(day_start) > 1:
            for t in day_start[:-1]:
                ax.axvline(x=t, color = 'gray', linestyle='--', linewidth=1, alpha=0.7)
        plt.show()
        cc2plt.savefig(tempdir + 'control_clicks.png')
        
        lpymax = int(max(fd['lpic']))
        if lpymax < 20:
            lpyticks = list(range(lpymax+1))
            lpymax += 0.2
        elif lpymax < 30: 
            lpyticks = list(range(0,lpymax+1,2))
            lpymax += 0.4
        else:
            lpyticks = list(range(0,lpymax+1,4))
            lpymax += 0.8
        lpplt = plt.figure(figsize=figsize)
        plt.scatter(fd['lx'], fd['lpic'], color='pink')
        plt.ylim(0,lpymax)
        plt.yticks(lpyticks)
        plt.xlim(0,xmax)
        plt.xticks(xticks)
        plt.title('Pokes following light onset (10s)')
        plt.xlabel('Trial')
        plt.ylabel('Count')
        ax = plt.gca()
        ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        if len(day_start) > 1:
            for t in day_start[:-1]:
                ax.axvline(x=t, color = 'gray', linestyle='--', linewidth=1, alpha=0.7)
        plt.show()
        lpplt.savefig(tempdir + 'trial_pokes_10s.png')
        
        ppspymax = np.abs(round(max(fd['ppsp']), 2))
        ppspyticks = list(range(int((ppspymax+0.05)*-100), int((ppspymax+0.05)*100), 2))
        ppspplt = plt.figure(figsize=figsize)
        plt.bar(bar_lab_x, fd['ppsp'], barw, color='pink')
        plt.ylim(-ppspymax,ppspymax)
        plt.yticks([i*0.01 for i in ppspyticks])
        plt.xlim(0, len(keys))
        plt.title('Pre/post stimulus port time difference')
        plt.ylabel('Percent difference (post - pre)')
        ax = plt.gca()
        ax.spines['bottom'].set_position(('data', ppspyticks[0]*0.01))
        ax.set_xticks(bar_lab_x)
        ax.set_xticklabels(bar_labs)
        ax.set_xlabel('Day')
        plt.axhline(0, color = 'black', linewidth=1)
        plt.show()
        ppspplt.savefig(tempdir + 'pre_post_time.png')