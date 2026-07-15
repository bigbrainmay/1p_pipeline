def cond_plots(gd_arr, count_dst=None, bar_dst=None):
    
    import matplotlib.pyplot as plt
    
    barx = [i+0.5 for i in range(len(gd_arr))]
    barw = 0.2
    
    allcount=[]
    allcount2=[]
    types=[]
    timebary=[]
    sess_lines=[]
    
    for i,gd in enumerate(gd_arr):
        types.extend(gd['Types'])
        allcount.extend(gd['lpi20']['count'])
        timebary.append(gd['l20d'])
        sess_lines.append(int(sum(sess_lines)+gd['nTrials']))
        
    print(sess_lines)
    allx = [i for i,t in enumerate(types) if t==1]
    countx=[allx[i] for i,c in enumerate(allcount) if c!=0]
    county=[c for c in allcount if c!=0]
    binary_y=[1 for c in allcount]
    
    pplt=plt.figure(figsize=(6,4))
    plt.scatter(countx, county, color='black')
    plt.ylim(0, max(county)+1)
    plt.yticks([i for i in range(0,max(county)+5, 5)])
    plt.xlim(0, len(types))
    plt.xticks([i for i in range(0, len(types)+5, 5)])
    plt.xlab('Conditioning trial')
    plt.ylab('Lick count')
    ax=plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in sess_lines[:-1]:
        ax.axvline(x=l, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    pplt.show()
    
    if count_dst:
        pplt.savefig(count_dst+'.png')
    
    bplt=plt.figure(figsize=(6,4))
    plt.scatter(countx, binary_y, color='black')
    plt.ylim(0, 1.2)
    plt.yticks([0,1])
    plt.xlim(0, len(types))
    plt.xticks([i for i in range(0, len(types)+5, 5)])
    plt.xlab('Conditioning trial')
    plt.ylab('Nosepoked trial (binary)')
    ax=plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in sess_lines[:-1]:
        ax.axvline(x=l, color='gray', linestyle='--', linewidth=1, alpha=0.7)
     
    bplt.show()
    
    if count_dst:
        bplt.savefig(count_dst+'_binary.png')
    
    
    tplt=plt.figure(figsize=(6,4))
    plt.bar(barx, timebary, barw, color='black')
    if min(timebary) > 0:
        plt.ylim(0, max(timebary)+1)
        plt.yticks([i for i in range(0,int(max(timebary)+6), 5)])
    else:
        plt.ylim(min(timebary)-1, max(timebary)+1)
        plt.yticks(i for i in range(min(timebary)-3, int(max(timebary)+2), 2))
    plt.xlim(0, len(gd_arr))
    plt.xticks(barx, [i+1 for i in range(len(barx))])
    plt.xlab('Conditioning day')
    plt.ylab('Port time difference pre/post CS onset')
    ax = plt.gca()
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='black', linewidth=1)
    
    tplt.show()
    
    if bar_dst:
        pplt.savefig(bar_dst+'.png')
        