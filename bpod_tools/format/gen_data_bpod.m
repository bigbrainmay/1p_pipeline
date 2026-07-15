function gen_data = gen_data_bpod(sesspath, pyfunc)
arguments
    sesspath string
    pyfunc logical = true
end

if exist(fullfile(sesspath, 'raw_bpod.mat'), 'file')
    raw_bpod = load(fullfile(sesspath, 'raw_bpod.mat'));
else
    warning('File does not contain raw_bpod.mat file: %s', sesspath);
    return
end

raw_bpod = raw_bpod.raw_bpod;

desc = raw_bpod.desc;
nTrials = raw_bpod.nTrials;
starts = raw_bpod.TrialStartTimestamp(:) - raw_bpod.TrialStartTimestamp(1);
stops = raw_bpod.TrialEndTimestamp(:) - raw_bpod.TrialStartTimestamp(1);

if numel(starts)+1 == nTrials
    nTrials = nTrials-1;
end

stateTable = table('Size', [0, 4], ...
    'VariableTypes', {'string', 'double', 'double', 'double'}, ...
    'VariableNames', {'Name', 'Start', 'Stop', 'Total'});
eventTable = table('Size', [0, 2], 'VariableTypes', {'string', 'double'}, ...
    'VariableNames', {'Name', 'Time'});
for i = 1 : nTrials
    trialStruct = struct;
    StateStruct = raw_bpod.RawEvents.Trial{1, i}.States;
    trialStruct.States.Names = fieldnames(StateStruct);
    trialStruct.States.Starts = structfun(@(x) x(1), StateStruct) + starts(i);
    trialStruct.States.Stops = structfun(@(x) x(2), StateStruct) + starts(i);
    trialStruct.States.Table = table(trialStruct.States.Names, ...
        trialStruct.States.Starts, trialStruct.States.Stops, ...
        trialStruct.States.Stops - trialStruct.States.Starts, ...
        'VariableNames', {'Name', 'Start', 'Stop', 'Total'});
    stateTable = [stateTable; trialStruct.States.Table];
    EventStruct = raw_bpod.RawEvents.Trial{1, i}.Events;
    trialStruct.Events.Names = fieldnames(EventStruct)';
    trialStruct.Events.Counts = structfun(@(x) length(x), EventStruct)';
    trialStruct.Events.Cols = vertcat(arrayfun(@(x) ...
        repmat(trialStruct.Events.Names(x), ...
        trialStruct.Events.Counts(x), 1), ...
        1: numel(trialStruct.Events.Names), 'UniformOutput', false));
    trialStruct.Events.Cols = vertcat(trialStruct.Events.Cols{:});
    trialStruct.Events.Times = struct2cell(EventStruct);
    trialStruct.Events.Times = [trialStruct.Events.Times{:}]' + starts(i);
    trialStruct.Events.Table = table(trialStruct.Events.Cols, ...
        trialStruct.Events.Times, ...
        'VariableNames', {'Name', 'Time'});
    eventTable = [eventTable; trialStruct.Events.Table];
end

stateTable(isnan(stateTable.Start), :) = [];

stateTable = sortrows(stateTable, 'Start');
eventTable = sortrows(eventTable, 'Time');

portTable = pokes(eventTable, stops(end));

stimTimes = stateTable(ismember(stateTable.Name, ["Tone", "Audio", "Light"]), :);

gen_data = struct( ...
        'stateTable', stateTable, ...
        'eventTable', eventTable, ...
        'desc', desc, ...
        'nTrials', nTrials, ...
        'trialStarts', starts, ...
        'trialStops', stops);

if isfield(raw_bpod, 'ProbeTypes')
    nProbes = numel(raw_bpod.ProbeTypes);
    probeStarts = starts(1:nProbes);
    probeStops = stops(1:nProbes);
    gen_data.nTrials = nTrials-nProbes;
    gen_data.trialStarts = starts(nProbes+1:end);
    gen_data.trialStops = stops(nProbes+1:end);
    gen_data.nProbes = nProbes;
    gen_data.probeTypes = raw_bpod.ProbeTypes;
    gen_data.probeStarts = probeStarts;
    gen_data.probeStops = probeStops;
end

if ~isempty(portTable)
    concatTable = [stateTable; portTable];
    concatTable = sortrows(concatTable, 'Start');
    gen_data.portTable = portTable;
    gen_data.concatTable = concatTable;
else
    gen_data.concatTable = stateTable;
end

if ~isempty(stimTimes)
    gen_data.stimTimes = stimTimes;
end

if isfield(raw_bpod, 'TrialTypes')
    types = raw_bpod.TrialTypes;
    if numel(types) > nTrials
        gen_data.Types = types(1:nTrials);
    else
        gen_data.Types = types;
    end
end

if isfield(raw_bpod, 'LightTimer')
    gen_data.lightTimer = raw_bpod.LightTimer;
end

if pyfunc == true
    gen_data = mat_to_py(gen_data);
end