

def pokes_in_range(gen_data, state, duration = 10, onset = True, delay=None, trialType = None, for_shuffle = False):
    
    """

    Parameters
    ----------
    gen_data : dict
        Gen_data dictionary created from some Bpod session(s). 
    state : string
        Name of desired state to find pokes within some related time range.
    onset : bool
        Determines whether to use the onset or offset of the state as a basis 
        for time range.
    duration : float or int
        Length (in seconds) of time range which is analyzed for pokes.
    delay : float or int, optional
        Indicates delay (in seconds) from onset/offset for start of time range.
        The default is None.
    trialType : int, optional
        Integer key indicating to only analyze specific trial types for pokes.
        The default is None.

    Returns
    -------
    Dict with fields 'range', 'times', 'latency', 'count', 'totalTime', 'desc', 'state', and 'duration'. 
    The first five fields are lists with an element for each instance of the state analyzed. 
    Elements contain the time frame start/stop, poke times, poke latencies from the 
    beginning of time frame, total poke count, and total time spent poking, 
    respectively, for each given instance of the state. 'desc' is the same 'desc' 
    field from the gen_data dict. 'state' and 'duration' are the parameters provided from the 
    function call. 'trialType' will be added as a field if the parameter is filled as well.
    
    """
    desc = gen_data['desc']
    
    ct = gen_data['concatTable']
    
    
    if isinstance(state, list):
        tt = ct
        st = state
    else:
    
        if trialType:
            ty = gen_data['Types']
            ti = gen_data['trialStarts']
            tf = gen_data['trialStops']
            
            # list of tuples with start/stop times for each trial of desired type
            if 'nProbes' in gen_data and state == 'Light':
                ty = gen_data['probeTypes']
                ti = gen_data['probeStarts']
                tf = gen_data['probeStops']
            
            tr = [(i,f) for t,i,f in zip(ty,ti,tf) if t == trialType] 
            
            # subset of concatTable only containing states for designated trial type
            tt = [s for s in ct if any([(s['Start'] >= t[0] and s['Stop'] <= t[1]+30) for t in tr])]
            
        else:
            tt = ct
        
        if onset:
            # start times for desired state
            st = [s['Start'] for s in tt if s['Name'] == state]
        else:
            st = [s['Stop'] for s in tt if s['Name'] == state]
        
        if delay:
            st = [s + delay for s in st]
            if for_shuffle:
                st = [s-tf[-1] for s in st if s>tf[-1]]
    
    # pokeTable subset for just pokes during desired trial types
    pt = [(s['Start'],s['Stop']) for s in tt if s['Name'] == '3' or s['Name'] == '1']
    
    # time window to look for pokes for each state instance
    r = [(i,i+duration) for i in st]
    
    t = [[s for s in pt if (s[0] > i[0] and s[0] < i[1])] for i in r]
    
    l = [[s[0]-j[0] for s in t[i]] for i,j in enumerate(r)]
    
    c = [len(s) for s in t]
    
    tp = [0 if not p else sum([s[1]-s[0] for s in p]) for p in t]
    
    tpt = sum(tp)
    
    s = sum([i[1]-i[0] for i in r])
    
    n = len([i for i in c if i != 0])
    
    if for_shuffle:
        pokes= {'range': r, 'count': c, 'sumtp': tpt}
    
    else:
        pokes = {'range': r, 'times': t, 'latency': l, 'count': c, 'responses': n, 'totalPoke': tp, 
                 'sumtp': tpt, 'totalTime': s, 'desc': desc, 'state': state, 'duration': duration}
    if trialType:
        pokes['trialType'] = trialType
    
    return pokes