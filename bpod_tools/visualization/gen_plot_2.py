def gen_plot_2(val_arrs_arr, title='', ylab='', xlabs=None, dst=None):
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os   

    x = [i+1 for i in range(len(val_arrs_arr))]
    ymin = 0
    ymax = int(max([max(i) for i in val_arrs_arr]))
    xmin = 0
    xmax = len(x)+1
    if ymax < 20:
        yticks = [i for i in range(ymax+2)]
    elif ymax < 40:
        yticks = [i for i in range(0, ymax+2, 2)]
    else:
        yticks = [i for i in range(0, ymax+2, 5)]
        
    if not xlabs:
        xlabs = [i+1 for i in range(len(x))]

    pplt = plt.figure(figsize=(6,4))
    plt.boxplot(val_arrs_arr, positions=x, labels=xlabs, widths=0.25)
    plt.scatter(x, [np.nanmedian(v) for v in val_arrs_arr], color='red', zorder=10)
    
    for i,v in enumerate(val_arrs_arr):
        jitter = np.random.normal(0,0.03,len(v))
        scatterx = ((x[i]+0.25)*np.ones(len(v))+jitter)
        plt.scatter(scatterx, v, color='black')
    
    plt.ylim(ymin, ymax+1)
    plt.xlim(xmin, xmax)
    plt.yticks(yticks)
    plt.yticks(yticks)
    ax = plt.gca()
    ax.spines['bottom'].set_position('zero'), ax.spines['left'].set_position('zero')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    
    
    plt.title(title)
    plt.ylabel(ylab)
    plt.xlabel(xlabs)
    
    plt.show()
    
    if dst:
        pplt.savefig(dst)
    