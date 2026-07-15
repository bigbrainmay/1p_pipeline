def merge_dicts(*dicts):
    if not dicts:
        return {}

    merged = {}

    for k in dicts[0].keys():
        values = [d[k] for d in dicts]
        if all(isinstance(v, int) for v in values): sum(values)
        
        first_type = type(values[0])
        if all(isinstance(v, list) for v in values):
            merged[k] = sum(values, [])
        elif all(isinstance(v, set) for v in values):
            merged[k] = set().union(*values)
        elif all(isinstance(v, first_type) for v in values):
            merged[k] = values
        else:
            raise TypeError(f"Incompatible types for key '{k}': {[type(v) for v in values]}")
    return merged