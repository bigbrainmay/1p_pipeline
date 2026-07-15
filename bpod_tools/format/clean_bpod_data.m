function clean_bpod_data(path)

gen_data = gen_data_bpod(path);
subj = gen_data.desc.subject;
sess = gen_data.desc.session;
filename = sprintf("%s_%s_cleaned_data_bpod.mat", subj, sess);
dst = fullfile(path, filename);

if ~isfield(gen_data, 'portTable')
    msg = "No poke data for this session; data cleaning irrelevant.";
    save(dst, "msg");
    return;
end

portTable = gen_data.portTable;
portStarts = portTable.Start;
stimTimes = gen_data.stimTimes;
stimStarts = stimTimes.Start;
stimStops = stimTimes.Stop;
stimTotals = stimTimes.Total;
pokeInds = arrayfun(@(x, y) find(portStarts > x & portStarts < y), stimStarts, stimStops, ...
            'UniformOutput', false);
pokeInds15 = arrayfun(@(x, y) find(portStarts > x & portStarts < y+15), stimStarts, stimStops, ...
            'UniformOutput', false);
pokeCounts = cellfun(@numel, pokeInds);
boolPokes = pokeCounts > 0;

cleaned_data_bpod = struct('gen_data', {gen_data}, 'stimTotals', {stimTotals}, ...
'pokeInds', {pokeInds}, 'pokeCounts', {pokeCounts});

if contains(path, ["\ConditioningJL", "/ConditioningJL"]) 
    if isfield(gen_data, 'lightTimer')
        lt = gen_data.lightTimer;
        pokeLatencies = arrayfun(@(x) (lt * (x == lt)) + ...
            (round(x,4) * (x~=lt)), stimTotals);
        pokeLatencies(pokeLatencies == round(lt,4)) = nan;
    else
        pokeLatencies = stimTotals;
    end
    cleaned_data_bpod.pokeLatencies = pokeLatencies;
end

if contains(path, ["PreconditioningJL", "Test"])
    
    types = gen_data.Types;
    plusTrials = types == 1;
    minusTrials = types == 2;

    
    pokeTimeIn = nan(1, numel(pokeInds));
    pokeLatencies = pokeTimeIn;
    pokeLatencies15 = pokeTimeIn;
    for p = 1 : numel(pokeInds)
        if ~isempty(pokeInds{p})
            pokeTimeIn(p) = sum(portTable.Total(pokeInds{p}));
            pokeLatencies(p) = portStarts(pokeInds{p}(1)) - stimStarts(p);
            pokeLatencies15(p) = portStarts(pokeInds15{p}(1))-stimStarts(p);
        end
    end

    cleaned_data_bpod.pokeTimeIn = pokeTimeIn;
    cleaned_data_bpod.pokeLatencies = pokeLatencies;
    cleaned_data_bpod.pokeLatencies15 = pokeLatencies15;

    if contains(path, "Test")

        plusStimTotal = sum(stimTimes.Total(plusTrials));
        minusStimTotal = sum(stimTimes.Total(minusTrials));
        plusPokes = sum(boolPokes(plusTrials));
        minusPokes = sum(boolPokes(minusTrials));
        plusPokeTimes = pokeTimeIn(plusTrials);
        plusPokeTimes = plusPokeTimes(~isnan(plusPokeTimes));
        plusPokeTotal = sum(plusPokeTimes);
        minusPokeTimes = pokeTimeIn(minusTrials);
        minusPokeTimes = minusPokeTimes(~isnan(minusPokeTimes));
        minusPokeTotal = sum(minusPokeTimes);

        dr = (plusPokes-minusPokes)/(plusPokes+minusPokes);
        plusPercIn = plusPokeTotal/plusStimTotal;
        minusPercIn = minusPokeTotal/minusStimTotal;

    elseif contains(path, "PreconditioningJL")
        concatTimes = gen_data.stimTimes; 
        lightInds = find(concatTimes.Name == "Light");
        concatTimes.Stop(lightInds-1) = concatTimes.Stop(lightInds); 
        concatTimes(lightInds, :) = [];
        concatTimes.Name = types';
        concatTimes.Total = concatTimes.Stop - concatTimes.Start;
        portStarts = gen_data.portTable.Start;
        concatPokeInds = arrayfun(@(x, y) find(portStarts(portStarts > x & portStarts <= y)), ...
                            concatTimes.Start, concatTimes.Stop, 'UniformOutput', false);
        concatPokes = cellfun(@numel, concatPokeInds);
        concatPokeLatencies = nan(1, numel(concatPokeInds));
        concatPokeTimeIn = nan(1, numel(concatPokeInds));
        for p = 1 : numel(concatPokeInds)
            if isempty(concatPokeInds{p})
                concatPokeLatencies(p) = nan;
                concatPokeTimeIn(p) = nan;
            else
                concatPokeLatencies(p) = portStarts(concatPokeInds{p}(1)) - concatTimes.Start(p);
                concatPokeTimeIn(p) = sum(concatTimes.Total(concatPokeInds{p})); 
            end
        end

        concatBoolPokes = ~isnan(concatPokeLatencies);
        cleaned_data_bpod.concatTimes = concatTimes;
        cleaned_data_bpod.concatPokeInds = concatPokeInds;
        cleaned_data_bpod.concatPokes = concatPokes;
        cleaned_data_bpod.concatPokeLatencies = concatPokeLatencies;
        cleaned_data_bpod.concatPokeTimeIn = concatPokeTimeIn;
        plusStimTotal = sum(concatTimes.Total(plusTrials));
        minusStimTotal = int16(sum(concatTimes.Total(minusTrials)));
        plusPokes = sum(concatBoolPokes(plusTrials));
        minusPokes = sum(concatBoolPokes(minusTrials));
        plusPokeTimes = concatPokeTimeIn(plusTrials);
        plusPokeTimes = plusPokeTimes(~isnan(plusPokeTimes));
        plusPokeTotal = sum(plusPokeTimes);
        minusPokeTimes = concatPokeTimeIn(minusTrials);
        minusPokeTimes = minusPokeTimes(~isnan(minusPokeTimes));
        minusPokeTotal = sum(minusPokeTimes);

        dr = (plusPokes-minusPokes)/(plusPokes+minusPokes);
        plusPercIn = plusPokeTotal/plusStimTotal;
        minusPercIn = minusPokeTotal/minusStimTotal;
    end   

    cleaned_data_bpod.plusTrials = plusTrials;
    cleaned_data_bpod.minusTrials = minusTrials;
    cleaned_data_bpod.plusStimTotal = plusStimTotal;
    cleaned_data_bpod.minusStimTotal = minusStimTotal;
    cleaned_data_bpod.plusPokes = plusPokes;
    cleaned_data_bpod.minusPokes = minusPokes;
    cleaned_data_bpod.plusPokeTotal = plusPokeTotal;
    cleaned_data_bpod.minusPokeTotal = minusPokeTotal;

    cleaned_data_bpod.dr = dr;
    cleaned_data_bpod.plusPercIn = plusPercIn;
    cleaned_data_bpod.minusPercIn = minusPercIn;

end

filename = sprintf("%s_%s_cleaned_data_bpod.mat", subj, sess);


dst = fullfile(path, filename);
save(dst, "cleaned_data_bpod");