def dat_file_to_obj(src):

    try:
        with open(src, 'r', encoding='utf-8') as f:
            obj = f.read()
    except UnicodeDecodeError:
        with open(src, 'rb') as f:
            obj = f.read()
    return obj