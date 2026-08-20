function pyObj = mat_to_py(path_or_obj)

    if exist("path_or_obj", 'file')
        obj = load(path_or_obj);
    else
        obj = path_or_obj;
    end

    if istable(obj)
        obj = table2struct(obj);
    end

    if isstruct(obj)
        pyObj = cell(1, numel(obj));
        fields = fieldnames(obj);
        for i = 1:numel(obj)
            tempStruct = struct();
            for f = 1:numel(fields)
                fname = fields{f};
                tempStruct.(fname) = mat_to_py(obj(i).(fname));
            end
            pyObj{i} = tempStruct;
        end
        if numel(obj) == 1
            pyObj = pyObj{1};
        end

    elseif iscell(obj)
        pyObj = cellfun(@mat_to_py, obj, 'UniformOutput', false);

    elseif isnumeric(obj) || islogical(obj) || ischar(obj)
        pyObj = obj;

    else
        % Convert unsupported MATLAB objects to string
        pyObj = char(string(obj));
    end
end