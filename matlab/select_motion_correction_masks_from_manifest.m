function select_motion_correction_masks_from_manifest(manifest_csv, output_root, varargin)
%SELECT_MOTION_CORRECTION_MASKS_FROM_MANIFEST Pick one mask per recording.
% Run in desktop MATLAB. Existing mask files are skipped unless Overwrite is true.
% Blank or unavailable neu_h5 paths fall back to the recording under output_root.

p = inputParser;
p.addParameter('Only', '');
p.addParameter('Overwrite', false);
p.addParameter('DatasetPath', '/data');
p.addParameter('NumFrames', 100);
p.parse(varargin{:});
opts = p.Results;
addpath(fileparts(mfilename('fullpath')));

T = readtable(manifest_csv, 'TextType', 'string', 'VariableNamingRule', 'preserve');
assert(all(ismember({'recording_id', 'neu_h5'}, T.Properties.VariableNames)), ...
    'Manifest must contain recording_id and neu_h5 columns.');

for i = 1:height(T)
    recording_id = strtrim(string(T.recording_id(i)));
    if ismissing(recording_id) || strlength(strtrim(recording_id)) == 0
        continue
    end
    if strlength(string(opts.Only)) > 0
        matches = recording_id == string(opts.Only);
        if ismember('mouse_id', T.Properties.VariableNames)
            mouse_id = strtrim(string(T.mouse_id(i)));
            matches = matches || (~ismissing(mouse_id) && mouse_id == string(opts.Only));
        end
        if ~matches
            continue
        end
    end

    manifest_h5 = strtrim(string(T.neu_h5(i)));
    local_h5 = fullfile(output_root, recording_id, 'neural', recording_id + "_miniscope.h5");
    input_h5 = manifest_h5;
    if ismissing(input_h5) || ~isfile(input_h5)
        input_h5 = local_h5;
    end
    mask_path = fullfile(output_root, recording_id, 'neural', recording_id + "_mc_mask.mat");
    if isfile(mask_path) && ~opts.Overwrite
        fprintf('%s: mask already exists, skipping\n', recording_id);
        continue
    end
    if ismissing(input_h5) || ~isfile(input_h5)
        fprintf(2, '%s: missing input H5, skipping\n', recording_id);
        fprintf(2, '  Manifest path: %s\n', manifest_h5);
        fprintf(2, '  Local path:    %s\n', local_h5);
        continue
    end

    fprintf('\nRecording %d/%d: %s\n', i, height(T), recording_id);
    fprintf('Input H5: %s\n', input_h5);
    select_motion_correction_mask(input_h5, mask_path, ...
        'DatasetPath', opts.DatasetPath, 'NumFrames', opts.NumFrames);
end
end
