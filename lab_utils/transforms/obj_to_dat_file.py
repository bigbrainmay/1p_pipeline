def obj_to_dat_file(obj, dst):

    if isinstance(obj, str):
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(obj)
    elif isinstance(obj, bytes):
        with open(dst, 'wb') as f:
            f.write(obj)
    else:
        print('Type incompatible with function')
        print('Compatible data types: str, bytes')