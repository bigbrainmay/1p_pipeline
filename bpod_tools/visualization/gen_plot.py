def gen_plot(pokes_in_range_dict, count = True, time = True, prop = True):
    
    """
    
    NOTE: Plots differentiating behavior between different trial types within the same
    protocol will have a separate function. 
    
    Parameters
    ----------
    pokes_in_range_dict : dict
        Dictionary with each entry containing an instance of the "pokes" dict 
        returned by pokes_in_range() from the bpod_tools.format module.
    count : boolean
        Indicates if plots regarding number of pokes are created.
        The default is True.
    latency : boolean
        Indicates if plots regarding latency of pokes are created.
        The default is True.
    time : boolean
        Indicates if plots regarding time poking per state are created.
        The default is True.
    prop : boolean
        Indicates if plots regarding proportion of poked instances are created.
        The default is True.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    vals = list(pokes_in_range_dict.values())
    
    
    root = r"Z:\Data\Julia\Jayce\data"
    
    descs = {s: v['desc'] for s,v in pokes_in_range_dict.items()}
    exp = list(set([v['experiment'] for v in descs.values()]))
    subjs = list(set([v['subject'] for v in descs.values()]))
    prots = list(set([v['protocol'] for v in descs.values()]))
    states = [v['state'] for v in vals]
    durations = [v['duration'] for v in vals]
    sds = list(set([(s,d) for s,d in zip(states, durations)]))
    
    ### NEED TO COMPLETE SECOND PARAMETER ###
    temp_dir = os.path.join(root, fr'SPC_14Day_1\figures\GG11_Conditioning_click_') 
    
    
    
    count = [p['count'] for p in vals]
    latency = [p['latency'] for p in vals]
    tp = [p['totalPoke'] for p in vals]

    ccount = [[c for c in ct if c > 0] for ct in count]
    clatency = [[l[0] for l in lat if l] for lat in latency]
    
    time = [sum(p)/len(tp) for p in tp]
    prop = [len(cc)/len(c) for c,cc in zip(count,ccount)]
    
    bx = list(range(len(vals)))
    bx = [x+1 for x in bx]
    barx = [x-0.5 for x in bx]
    barw = 0.2
    if len(bx) < 6:
        figsize = (6,4)
    else:
        figsize = (len(bx)+1, 4)
    ls = [len(s['range']) for s in vals]
    js = [np.random.normal(0,0.03,size=l) for l in ls]
    clls = [len(l) for l in clatency if l]
    cljs = [np.random.normal(0,0.03,size=cl) for cl in clls]
    sxs = [((b+0.25)*np.ones(l))+j for b,l,j in zip(bx,ls,js)]
    clsxs = [((b+0.25)*np.ones(cl))+cj for b,cl,cj in zip(bx,clls,cljs)]
    count_ymax = max(max(c) for c  in count)
    if count_ymax <= 15:
        count_yticks = list(range(int(count_ymax)+1))
    else:
        count_yticks = list(range(0,int(count_ymax)+1,2))
    if max(time) <= 2.5:
        time_ymax = round(max(time),1)
        time_yticks = [x*0.1 for x in range(int((time_ymax+0.1)*10))]
    else:
        time_ymax = np.ceil(max(time))
        time_yticks = list(range(int(time_ymax)+1))
    prop_ymax = 1.02
    prop_yticks = [x*0.1 for x in range(11)]
    ### NEED TO MANUALLY INDICATE ###
    count_title = 'Poke counts/stimulus'
    time_title = 'Time poking/stimulus'
    prop_title = 'Proportion of Poked stimuli'
    box_labels = ['Tone', 'Noise']# [f'{i+1}' for i in range(len(pokes_in_range_dict))]
    x_lab = 'Day'
            
    if count:
        pcplt = plt.figure(figsize=figsize)
        plt.boxplot(count, positions=bx, labels=box_labels, widths=0.25)
        plt.scatter(bx, [np.nanmedian(c) for c in count], color='red',zorder=10)
        for i,c in enumerate(count):
            plt.scatter(sxs[i], c, color='black')
        plt.title(count_title)
        plt.xlabel(x_lab)
        plt.ylabel('Count')
        plt.ylim(0,(count_ymax+0.3))
        plt.yticks(count_yticks)
        ax = plt.gca()
        ax.spines['bottom'].set_position('zero'), 
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        plt.show()
        # pcplt.savefig(temp_dir + 'count.png')
        
        # cleaned count (zeros excluded)
        pcplt = plt.figure(figsize=figsize)
        plt.boxplot(ccount, positions=bx, labels=box_labels, widths=0.25)
        plt.scatter(bx, [np.nanmedian(c) for c in ccount], color='red',zorder=10)
        for i,c in enumerate(ccount):
            plt.scatter(clsxs[i], c, color='black')
        plt.title(count_title)
        plt.xlabel(x_lab)
        plt.ylabel('Count (zeroes omitted)')
        plt.ylim(0,(count_ymax+0.3))
        plt.yticks(count_yticks)
        ax = plt.gca()
        ax.spines['bottom'].set_position('zero'), 
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        plt.show()
        # pcplt.savefig(temp_dir + 'clean_count.png')
            
    if time:
        ptplt = plt.figure(figsize=figsize)
        plt.bar(barx, time, barw, color='black')
        plt.title(time_title)
        plt.xlabel(x_lab)
        plt.ylabel('Time (s)')
        plt.xlim(0,max(bx))
        plt.ylim(0,time_ymax+0.1)
        plt.xticks(barx, box_labels)
        plt.yticks(time_yticks)
        ax = plt.gca()
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        plt.show()
        # ptplt.savefig(temp_dir + 'time.png')
    
    if prop:
        ppplt = plt.figure(figsize=figsize)
        plt.bar(barx, prop, barw, color='black')
        plt.title(prop_title)
        plt.xlabel(x_lab)
        plt.ylabel('Total pokes/day')
        plt.ylim(0,prop_ymax+0.3)
        plt.xlim(0,max(bx))
        plt.xticks(barx, box_labels)
        plt.yticks(prop_yticks)
        ax = plt.gca()
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        plt.show()
        # ppplt.savefig(temp_dir + 'prop.png')