function clickTable = clicks(eventTable)

clickTable = table('Size', [0, 4], ...
    'VariableTypes', {'string', 'double', 'double', 'double'}, ...
    'VariableNames', {'Name', 'Start', 'Stop', 'Total'});

clickStart = 'GlobalTimer1_Start';
clickEnd = 'GlobalTimer1_End';
startMask = ismember(eventTable.Name, clickStart);
endMask = ismember(eventTable.Name, clickEnd);
clickStarts = eventTable.Time(startMask);
clickEnds = eventTable.Time(endMask);

if isempty(clickStarts)
    clickTable = {};
    return
end

clickTotals = clickEnds-clickStarts;
name_col = repmat({'click'}, numel(clickStarts), 1);
clickTable = table(name_col, clickStarts, clickEnds, clickTotals, ...
    'VariableNames', {'Name', 'Start', 'Stop', 'Total'});


