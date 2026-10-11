function summary = run_motion_correction_one_record(input_h5_path, output_h5_path, recording_id, varargin)
%RUN_MOTION_CORRECTION_ONE_RECORD Motion-correct one pipeline miniscope H5.
%
% This stage intentionally writes a separate motion-corrected movie before
% EXTRACT preprocessing. If PrefilterEnable is true, a chunked spatial
% bandpass movie is first written and then used as the NormCorre input.
% EXTRACT's later preprocess_save stage is still separate and writes *_final.h5.

opts = parse_options(varargin{:});

input_h5_path = char(input_h5_path);
output_h5_path = char(output_h5_path);
recording_id = char(recording_id);
dataset_path = char(opts.DatasetPath);
prefilter_h5_path = char(opts.PrefilterH5Path);

output_parent = fileparts(output_h5_path);
if ~exist(output_parent, 'dir')
    mkdir(output_parent);
end

fprintf('\n=== Motion correction %s ===\n', recording_id);
fprintf('Input H5:        %s\n', input_h5_path);
fprintf('Output H5:       %s\n', output_h5_path);

if exist(output_h5_path, 'file') && ~opts.Overwrite
    fprintf('Using existing motion-corrected H5: %s\n', output_h5_path);
    summary = make_summary(recording_id, input_h5_path, output_h5_path, 'exists');
    return
end

if strcmp(input_h5_path, output_h5_path)
    error('Input and output H5 paths must be different: %s', input_h5_path);
end

mc_input_h5_path = input_h5_path;
if opts.PrefilterEnable
    if missing_text(prefilter_h5_path)
        [folder, name, ext] = fileparts(output_h5_path);
        prefilter_h5_path = fullfile(folder, [name '_prefilter' ext]);
    end
    fprintf('Prefilter H5:    %s\n', prefilter_h5_path);
    chunked_spatial_bandpass_to_h5(input_h5_path, prefilter_h5_path, opts);
    mc_input_h5_path = prefilter_h5_path;
else
    fprintf('Prefilter H5:    disabled\n');
end

if exist(output_h5_path, 'file')
    delete(output_h5_path);
end

config_mc = [];
config_mc = get_defaults_mc(config_mc);
config_mc.get_mask = opts.GetMask;

mc_input = [mc_input_h5_path ':' dataset_path];
mc_output = [output_h5_path ':' dataset_path];
fprintf('NormCorre input: %s\n', mc_input);
fprintf('NormCorre out:   %s\n', mc_output);
run_normcorre_pipeline(mc_input, mc_output, config_mc);

if opts.PrefilterEnable && ~opts.KeepPrefilter && exist(prefilter_h5_path, 'file')
    delete(prefilter_h5_path);
end

summary = make_summary(recording_id, input_h5_path, output_h5_path, 'completed');
end


function chunked_spatial_bandpass_to_h5(input_h5_path, output_h5_path, opts)
dataset_path = char(opts.DatasetPath);

if exist(output_h5_path, 'file') && ~opts.Overwrite
    fprintf('Using existing prefilter H5: %s\n', output_h5_path);
    return
end

if exist(output_h5_path, 'file')
    delete(output_h5_path);
end

info = h5info(input_h5_path, dataset_path);
dims = info.Dataspace.Size;
height = dims(1);
width = dims(2);
n_frames_total = dims(3);
chunk_size = min(opts.PrefilterChunkSize, n_frames_total);

fprintf('Writing chunked spatial bandpass movie: %s\n', output_h5_path);
for start_frame = 1:chunk_size:n_frames_total
    n_frames = min(chunk_size, n_frames_total - start_frame + 1);
    fprintf('Prefilter frames %d-%d / %d\n', ...
        start_frame, start_frame + n_frames - 1, n_frames_total);

    chunk = single(h5read( ...
        input_h5_path, ...
        dataset_path, ...
        [1, 1, start_frame], ...
        [height, width, n_frames]));

    filtered = spatial_bandpass( ...
        chunk, ...
        opts.PrefilterAvgCellRadius, ...
        opts.PrefilterSpatialHighpassCutoff, ...
        opts.PrefilterSpatialLowpassCutoff, ...
        opts.PrefilterUseGpu);
    filtered = single(gather(filtered));

    if start_frame == 1
        h5create( ...
            output_h5_path, ...
            dataset_path, ...
            [height, width, n_frames_total], ...
            'Datatype', 'single', ...
            'ChunkSize', [height, width, min(100, n_frames_total)]);
    end

    h5write( ...
        output_h5_path, ...
        dataset_path, ...
        filtered, ...
        [1, 1, start_frame], ...
        [height, width, n_frames]);

    clear chunk filtered
end
end


function opts = parse_options(varargin)
p = inputParser;
p.addParameter('DatasetPath', '/data');
p.addParameter('Overwrite', false);
p.addParameter('PrefilterEnable', true);
p.addParameter('PrefilterH5Path', '');
p.addParameter('PrefilterChunkSize', 1000);
p.addParameter('PrefilterAvgCellRadius', 6);
p.addParameter('PrefilterSpatialHighpassCutoff', 5);
p.addParameter('PrefilterSpatialLowpassCutoff', 2);
p.addParameter('PrefilterUseGpu', 0);
p.addParameter('KeepPrefilter', false);
p.addParameter('GetMask', 1);
p.parse(varargin{:});
opts = p.Results;

opts.Overwrite = logical_flag(opts.Overwrite);
opts.PrefilterEnable = logical_flag(opts.PrefilterEnable);
opts.PrefilterChunkSize = numeric_value(opts.PrefilterChunkSize);
opts.PrefilterAvgCellRadius = numeric_value(opts.PrefilterAvgCellRadius);
opts.PrefilterSpatialHighpassCutoff = numeric_value(opts.PrefilterSpatialHighpassCutoff);
opts.PrefilterSpatialLowpassCutoff = numeric_value(opts.PrefilterSpatialLowpassCutoff);
opts.PrefilterUseGpu = numeric_flag(opts.PrefilterUseGpu);
opts.KeepPrefilter = logical_flag(opts.KeepPrefilter);
opts.GetMask = numeric_flag(opts.GetMask);
end


function summary = make_summary(recording_id, input_h5_path, output_h5_path, status)
summary = struct();
summary.recording_id = recording_id;
summary.input_h5_path = input_h5_path;
summary.output_h5_path = output_h5_path;
summary.status = status;
end


function value = numeric_value(value)
if ischar(value) || isstring(value)
    value = str2double(value);
end
end


function value = numeric_flag(value)
if ischar(value) || isstring(value)
    value = str2double(value);
end
value = double(logical(value));
end


function value = logical_flag(value)
if ischar(value) || isstring(value)
    text = lower(strtrim(char(value)));
    value = any(strcmp(text, {'1', 'true', 'yes', 'y', 'on'}));
else
    value = logical(value);
end
end


function tf = missing_text(value)
if isstring(value)
    value = char(value);
end
if isempty(value)
    tf = true;
elseif ischar(value)
    text = lower(strtrim(value));
    tf = any(strcmp(text, {'', 'nan', 'none', 'null'}));
else
    tf = false;
end
end
