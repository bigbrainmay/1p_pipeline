def paradigm_plots(gd_arr, dst=None):
    
    import matplotlib.pyplot as plt
    
    pc = [s for s in gd_arr if s['desc']['protocol']=='Preconditioning']
    c = gd_arr[len(pc):-1]
    t = gd_arr[-1]
    
    pd=[s['ld']for s in pc]+[s['ld']for s in c]+[t['ld'],t['td']]
    pd20=[s['l20d']for s in pc]+[s['l20d']for s in c]+[t['l20d'],t['t20d']]
    
    tdr=[s['ltdr']for s in pc]+[s['ltdr']for s in c]+[t['ltdr'],t['atdr']]
    cdr=[s['lcdr']for s in pc]+[s['lcdr']for s in c]+[t['lcdr'],t['acdr']]
    bdr=[s['lbdr']for s in pc]+[s['lbdr']for s in c]+[t['lbdr'],t['abdr']]
    tdr20=[s['ltdr20']for s in pc]+[s['ltdr20']for s in c]+[t['ltdr20'],t['atdr20']]
    cdr20=[s['lcdr20']for s in pc]+[s['lcdr20']for s in c]+[t['lcdr20'],t['acdr20']]
    bdr20=[s['lbdr20']for s in pc]+[s['lbdr20']for s in c]+[t['lbdr20'],t['abdr20']]
    
    nTrials=[0]
    nPairs=[0]
    types=[]
    count1s=[]
    time1s=[]
    count2s=[]
    time2s=[]
    count1s20=[]
    time1s20=[]
    count2s20=[]
    time2s20=[]
    tdrt=[]
    cdrt=[]
    bdrt=[]
    tdr20t=[]
    cdr20t=[]
    bdr20t=[]
    
    for s in gd_arr:
        if s['desc']['protocol']!='Test':
            nTrials.append(nTrials[-1]+s['nTrials'])
            nPairs.append(nPairs[-1]+len(s['ltdrt']))
            types.extend(s['Types'])
        count1s.extend(s['lpi']['count'])
        time1s.extend(s['lpi']['totalPoke'])
        count2s.extend(s['dpi']['count'])
        time2s.extend(s['dpi']['totalPoke'])
        count1s20.extend(s['lpi20']['count'])
        time1s20.extend(s['lpi20']['totalPoke'])
        count2s20.extend(s['dpi20']['count'])
        time2s20.extend(s['dpi20']['totalPoke'])
        tdrt.extend(s['ltdrt'])
        cdrt.extend(s['lcdrt'])
        bdrt.extend(s['lbdrt'])
        tdr20t.extend(s['ltdr20t'])
        cdr20t.extend(s['lcdr20t'])
        bdr20t.extend(s['lbdr20t'])
        if s['desc']['protocol']=='Test':
            types.extend(s['probeTypes']+s['Types'])
            count1s.extend(s['tpi']['count'])
            time1s.extend(s['tpi']['totalPoke'])
            count2s.extend(s['npi']['count'])
            time2s.extend(s['npi']['totalPoke'])
            count1s20.extend(s['tpi20']['count'])
            time1s20.extend(s['tpi20']['totalPoke'])
            count2s20.extend(s['npi20']['count'])
            time2s20.extend(s['npi20']['totalPoke'])
            tdrt.extend(s['atdrt'])
            cdrt.extend(s['acdrt'])
            bdrt.extend(s['abdrt'])
            tdr20t.extend(s['atdr20t'])
            cdr20t.extend(s['acdr20t'])
            bdr20t.extend(s['abdr20t'])
            probetl=nTrials[-1]+s['nProbes'] # probetl - probe trial line
            probepl=nPairs[-1]+len(s['atdrt']) # probepl - probe pairs line
    bin1s=[c!=0 for c in count1s]
    bin1s20=[c!=0 for c in count1s20]
    bin2s=[c!=0 for c in count2s]
    bin2s20=[c!=0 for c in count2s20]
    
    t1x=[i for i,t in enumerate(types) if t==1]
    t2x=[i for i,t in enumerate(types) if t==2]
    
    t1nzx=[t for i,t in enumerate(t1x) if bin1s[i]==1]
    count1nz=[c for c in count1s if c!=0]
    time1nz=[t for t in time1s if t!=0]
    t1nz20x=[t for i,t in enumerate(t1x) if bin1s20[i]==1]
    count1nz20=[c for c in count1s20 if c!=0]
    time1nz20=[t for t in time1s20 if t!=0]
    
    t2nzx=[t for i,t in enumerate(t2x) if bin2s[i]==1]
    count2nz=[c for c in count2s if c!=0]
    time2nz=[t for t in time2s if t!=0]
    t2nz20x=[t for i,t in enumerate(t2x) if bin2s20[i]==1]
    count2nz20=[c for c in count2s20 if c!=0]
    time2nz20=[t for t in time2s20 if t!=0]
    
    x=t1x+t2x
    px=[i for i in range(len(tdrt)+1)]
        
    cplt=plt.figure(figsize=(6,4))
    plt.scatter(t1x, count1s, color='pink', s=20, clip_on=False)
    plt.scatter(t2x, count2s, color='black', s=20, clip_on=False)
    plt.ylim(0,max(count1s+count2s))
    plt.xlim(0,max(x))
    plt.yticks([i for i in range(0,int(max(count1s+count2s)+5),5)])
    plt.xticks([i for i in range(0, int(max(x)+50), 50)])
    plt.xlabel('Trial')
    plt.ylabel('Lick count during stimulus')
    ax=plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in nTrials:
        ax.axvline(x=l, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=probetl, color='red', linestyle='--', linewidth=1)
    cplt.show()
    
    cnzplt=plt.figure(figsize=(6,4))
    plt.scatter(t1nzx, count1nz, color='pink', s=20, clip_on=False)
    plt.scatter(t2nzx, count2nz, color='black', s=20, clip_on=False)
    plt.ylim(0, max(count1nz+count2nz))
    plt.xlim(0,max(x))
    plt.yticks([i for i in range(0, int(max(count1nz+count2nz)+5), 5)])
    plt.xticks([i for i in range(0, int(max(x)+50), 50)])
    plt.xlabel('Trial')
    plt.ylabel('Lick count during stimulus')
    ax=plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in nTrials:
        ax.axvline(x=l, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=probetl, color='red', linestyle='--', linewidth=1)
    cnzplt.show()
    
    c20plt=plt.figure(figsize=(6,4))
    plt.scatter(t1x, count1s20, color='pink', s=20, clip_on=False)
    plt.scatter(t2x, count2s20, color='black', s=20, clip_on=False)
    plt.ylim(0,max(count1s20+count2s20))
    plt.xlim(0,max(x))
    plt.yticks([i for i in range(0,int(max(count1s20+count2s20)+5),5)])
    plt.xticks([i for i in range(0, int(max(x)+50), 50)])
    plt.xlabel('Trial')
    plt.ylabel('Lick count 20s post-stim. onset')
    ax=plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in nTrials:
        ax.axvline(x=l, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=probetl, color='red', linestyle='--', linewidth=1)
    c20plt.show()
    
    cnz20plt=plt.figure(figsize=(6,4))
    plt.scatter(t1nz20x, count1nz20, color='pink', s=20, clip_on=False)
    plt.scatter(t2nz20x, count2nz20, color='black', s=20, clip_on=False)
    plt.ylim(0, max(count1nz20+count2nz20))
    plt.xlim(0, max(x))
    plt.yticks([i for i in range(0,int(max(count1nz20+count2nz20)+5), 5)])
    plt.xticks([i for i in range(0, int(max(x)+50), 50)])
    plt.xlabel('Trial')
    plt.ylabel('Lick count 20s post-stim. onset')
    ax=plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in nTrials:
        ax.axvline(x=l, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=probetl, color='red', linestyle='--', linewidth=1)
    cnz20plt.show()
        
    tplt=plt.figure(figsize=(6,4))
    plt.scatter(t1x, time1s, color='pink', s=20, clip_on=False)
    plt.scatter(t2x, time2s, color='black', s=20, clip_on=False)
    plt.ylim(0, max(time1s+time2s))
    plt.xlim(0, max(x))
    plt.yticks([i for i in range(0, int(max(time1s+time2s)+2), 2)])
    plt.xticks([i for i in range(0, int(max(x)+50), 50)])
    plt.xlabel('Trial')
    plt.ylabel('Lick time during stimulus')
    ax=plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in nTrials:
        ax.axvline(x=l, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=probetl, color='red', linestyle='--', linewidth=1)
    tplt.show()
    
    tnzplt=plt.figure(figsize=(6,4))
    plt.scatter(t1nzx, time1nz, color='pink', s=20, clip_on=False)
    plt.scatter(t2nzx, time2nz, color='black', s=20, clip_on=False)
    plt.ylim(0, max(time1nz+time2nz))
    plt.xlim(0, max(x))
    plt.yticks([i for i in range(0, int(max(time1nz+time2nz)+2), 2)])
    plt.xticks([i for i in range(0, int(max(x)+50), 50)])
    plt.xlabel('Trial')
    plt.ylabel('Lick time during stimulus')
    ax=plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in nTrials:
        ax.axvline(x=l, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=probetl, color='red', linestyle='--', linewidth=1)
    tnzplt.show()
    
    t20plt=plt.figure(figsize=(6,4))
    plt.scatter(t1x, time1s20, color='pink', s=20, clip_on=False)
    plt.scatter(t2x, time2s20, color='black', s=20, clip_on=False)
    plt.ylim(0, max(time1s20+time2s20))
    plt.xlim(0, max(x))
    plt.yticks([i for i in range(0, int(max(time1s20+time2s20)+2), 2)])
    plt.xticks([i for i in range(0, int(max(x)+50), 50)])
    plt.xlabel('Trial')
    plt.ylabel('Lick time 20s post-stim. onset')
    ax=plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in nTrials:
        ax.axvline(x=l, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=probetl, color='red', linestyle='--', linewidth=1)
    t20plt.show()
    
    tnz20plt=plt.figure(figsize=(6,4))
    plt.scatter(t1nz20x, time1nz20, color='pink', s=20, clip_on=False)
    plt.scatter(t2nz20x, time2nz20, color='black', s=20, clip_on=False)
    plt.ylim(0, max(time1nz20+time2nz20))
    plt.xlim(0, max(x))
    plt.yticks([i for i in range(0, int(max(time1nz20+time2nz20)+2), 2)])
    plt.xticks([i for i in range(0, int(max(x)+50), 50)])
    plt.xlabel('Trial')
    plt.ylabel('Lick time 20s post-stim. onset')
    ax=plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in nTrials:
        ax.axvline(x=l, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=probetl, color='red', linestyle='--', linewidth=1)
    tnz20plt.show()
    
    bplt=plt.figure(figsize=(6,4))
    plt.scatter(t1x, bin1s, color='pink', s=20, clip_on=False)
    plt.scatter(t2x, bin2s, color='black', s=20, clip_on=False)
    plt.ylim(0, 1)
    plt.xlim(0, max(x))
    plt.yticks([0,1])
    plt.xticks([i for i in range(0, int(max(x)+50), 50)])
    plt.xlabel('Trial')
    plt.ylabel('Licked during stimulus')
    ax=plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in nTrials:
        ax.axvline(x=l, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=probetl, color='red', linestyle='--', linewidth=1)
    bplt.show()
    
    b20plt=plt.figure(figsize=(6,4))
    plt.scatter(t1x, bin1s20, color='pink', s=20, clip_on=False)
    plt.scatter(t2x, bin2s20, color='black', s=20, clip_on=False)
    plt.ylim(0, 1)
    plt.xlim(0, max(x))
    plt.yticks([0,1])
    plt.xticks([i for i in range(0, int(max(x)+50), 50)])
    plt.xlabel('Trial')
    plt.ylabel('Licked during stimulus')
    ax=plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in nTrials:
        ax.axvline(x=l, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=probetl, color='red', linestyle='--', linewidth=1) 
    b20plt.show()
    
    tdrplt=plt.figure(figsize=(6,4))
    plt.scatter(px, tdrt, color='black', s=20, clip_on=False)
    plt.ylim(-1,1)
    plt.xlim(0,max(px))
    plt.yticks([-1,0,1])
    plt.xticks([i for i in range(0, int(max(px)+10), 10)])
    plt.xlabel('Paired cues')
    plt.ylabel('Lick time disc. ratio')
    ax=plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in nPairs:
        ax.axvline(x=l, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=probepl, color='red', linestyle='--', linewidth=1)
    tdrplt.show()