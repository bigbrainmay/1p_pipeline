function portTable = pokes(eventTable, stop)

pokeList = unique(eventTable.Name(startsWith(eventTable.Name, 'Port')));

if isempty(pokeList)
    portTable = {};
    return
end
portNums = unique(cellfun(@(x) x(5), cellstr(pokeList)));

portTable = table('Size', [0, 4], ...
    'VariableTypes', {'string', 'double', 'double', 'double'}, ...
    'VariableNames', {'Name', 'Start', 'Stop', 'Total'});

for i = 1 : numel(portNums)
    portNum = portNums(i);
    portIn = sprintf("Port%sIn", portNum);
    portOut = sprintf("Port%sOut", portNum);
    
    mask = ismember(eventTable.Name, ...
        [portIn, portOut]);
    justPort = eventTable(mask, :);
    if strcmp(justPort.Name(1), portOut);
        justPort = justPort(2:end, :);
    end
    if strcmp(justPort.Name(end), portIn)
        justPort(end+1, :) = {portOut, stop};
    end
    
    badInds = [];
    for p = 1 : height(justPort)-1
        if strcmp(justPort.Name(p), justPort.Name(p+1))
            badInds = [badInds, p+1];
        end
    end
    justPort(badInds, :) = [];
    
    pokeThresh = 0.1;
    
    portIns = justPort.Time(ismember(justPort.Name,portIn), :);
    portOuts = justPort.Time(ismember(justPort.Name,portOut), :);
    
    diffs = portIns(2:end)-portOuts(1:end-1);
    valid_diffs = diffs > pokeThresh;
    
    while ~all(valid_diffs)
        
        maskOut = [diffs<pokeThresh; false];
        maskIn = [false; diffs<pokeThresh];
        
        portOuts(maskOut) = [];
        portIns(maskIn) = [];
    
        diffs = portIns(2:end)-portOuts(1:end-1);
        valid_diffs = diffs > pokeThresh;
    
    end 
    name_col = repmat({sprintf('%s',portNum)}, numel(portIns), 1);
    tempTable = table(name_col, portIns, portOuts, (portOuts - portIns), ...
                'VariableNames', {'Name', 'Start', 'Stop', 'Total'});
    
    portTable = [portTable; tempTable];
end

portTable = sortrows(portTable, 'Start');