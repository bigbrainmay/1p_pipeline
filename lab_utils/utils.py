###

def extract_matches(source, to_match, compiled = True, basename = True, search = False, rm_exts = False, is_bool = False):
    import os
    import re
    
    if basename:
        source = [os.path.basename(os.path.abspath(s)) for s in source]

    if rm_exts:
        source = [os.path.splitext(s)[0] for s in source]
        
    if compiled:
        if not isinstance(to_match, re.Pattern):
            print(f'Error in extract_matches: {to_match} is not type re.Pattern')
            return None
        if search:
            out =  [s for s in source if to_match.search(s)]
        else:
            out =  [s for s in source if to_match.match(s)]
    else:
        if search:
            out =  [s for s in source if re.search(to_match, s)]
        else:
            out =  [s for s in source if re.match(to_match, s)]

    if is_bool:
        return bool(out)
    else:
        return out
        
###

def nansem(data):
    import numpy as np
    sd = np.nanstd(data)
    n = len(data)
    sem = sd/np.sqrt(n)
    return sem

###

def sort_file_dates(dates):
    from datetime import datetime
    dates_sorted = sorted(dates, key=lambda d: datetime.strptime(d, "%M%d%Y"))
    return dates_sorted