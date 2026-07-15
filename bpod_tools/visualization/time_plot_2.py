def time_plot_2(val_arr, trialCounts_arr, rm_zeros=False, dst=None, title='', ylab='', xlab=''):
    
    import matplotlib.pyplot as plt
    
    val_len = len(val_arr)
    x = [i+1 for i in range(val_len)]
    
    if rm_zeros:
        x = [i for i,t in enumerate(val_arr) if t!=0]
        val_arr=[t for t in val_arr if t!=0]
    
    ymax = max(val_arr)+1
    ymin = 0
    xmin = 0
    xmax = val_len+2
    if xmax < 20:
        xticks = [i for i in range(xmax-1)]
    elif xmax < 40:
        xticks = [i for i in range(0, xmax-1, 2)]
    elif xmax < 80:
        xticks = [i for i in range(0, xmax-1, 5)]
    else:
        xticks = [i for i in range(0, xmax-1, 20)]
    if ymax < 20:
        yticks = [i for i in range(ymax+2)]
    elif ymax < 40:
        yticks = [i for i in range(0, ymax+2, 2)]
    else:
        yticks = [i for i in range(0, ymax+2, 5)]
    
    
    pplt = plt.figure(figsize=(6,4))
    plt.scatter(x, val_arr, color='black')
    plt.ylim(ymin,ymax)
    plt.yticks(yticks)
    plt.xlim(xmin, xmax)
    plt.xticks(xticks)
    ax = plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    for l in trialCounts_arr:
        ax.axvline(x=int(l), color='gray', linestyle='--', linewidth=1, alpha=0.7)
        
    plt.title(title)
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    
    pplt.show()
    
    if dst:
        pplt.savefig(dst)
    
