def prop_plot(val_arr, title='', ylab='', xlab='', colors=None, dst=None):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
        
    x = [(i/2)+0.5 for i in range(len(val_arr))]
    w = 0.2
    absmaxy = max(np.abs(val_arr))
    xmin = 0
    xmax = (len(x)/2)+0.25
    ymin=0
    if absmaxy < 0.1:
        ymax = 0.1
        yticks = [(i*0.02)-0.1 for i in range(11)]
    elif absmaxy < 0.25:
        ymax = 0.25
        yticks = [(i*0.05)-0.25 for i in range(11)]
    elif absmaxy < 0.5:
        ymax = 0.5
        yticks = [(i*0.05)-0.5 for i in range(21)]
    elif absmaxy < 1:
        ymax = 1
        yticks = [(i*0.1)-1 for i in range(21)]
    else:
        ymax = int(absmaxy+1)
        yticks = [i-ymax for i in range((ymax*2)+1)]
    xticks = ['' for i in x]
        
    pplt = plt.figure(figsize=(6,4))
    plt.bar(x, val_arr, w, color='black')
    plt.ylim(ymin, ymax)
    plt.yticks(yticks)
    plt.xlim(0,xmax)
    plt.xticks(x,xticks)
    ax = plt.gca()
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.spines['bottom'].set_visible(False)
    
    plt.ylabel(ylab)
    plt.xlabel(xlab)
    
    plt.show()
    
    if dst:
        pplt.savefig(dst)
    