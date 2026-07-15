def plot_protocol(prot_gd_arr):
    
    from bpod_tools import time_plot_2
    from bpod_tools import gen_plot_2
    
    if all(['lpi' in p and 'dpi' in p for p in prot_gd_arr]):
        
        genlabs=[f'Day {i+1}' for i in range(len(prot_gd_arr))]
        
        lightva = []
        lighttca = []
        lightvaa = []
        dimva = []
        dimtca = []
        dimvaa = []
        for i,p in enumerate(prot_gd_arr):
            lightva.extend(p['lpi']['count'])
            lighttca.append(len(p['lpi']['count']))
            lightvaa.append(p['lpi']['count'])
            dimva.extend(p['dpi']['count'])
            dimtca.append(len(p['dpi']['count']))
            dimvaa.append(p['dpi']['count'])
            
        time_plot_2(lightva, lighttca, title='Flashing light poke counts', ylab='Count', xlab='Trial')
        time_plot_2(dimva, dimtca, title='Dim light poke counts', ylab='Count', xlab='Trial')
        gen_plot_2(lightvaa, title='Flashing light poke counts', ylab='Count', xlabs=genlabs)
        gen_plot_2(dimvaa, title='Dim light poke counts', ylab='Count', xlabs=genlabs)
        