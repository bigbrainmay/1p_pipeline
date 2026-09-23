function summary = run_extract_one_record(neu_h5_path, output_dir, recording_id, varargin)
%RUN_EXTRACT_ONE_RECORD Run EXTRACT on one pipeline-generated miniscope H5.
%
% This wraps the Schnitzer lab EXTRACT workflow used in the notebook:
%   1. preprocess_save('<input>.h5:/data', config)
%   2. extractor({'<input>_final.h5', '/data'}, config)
%   3. save <recording_id>_extract_output_unsorted.mat for Python import and
%      as the source for ActSort/manualActSort curation
%
% The saved MAT file contains:
%   - output: raw EXTRACT output structure
%
% ActSort/manualActSort labels are still a manual curation checkpoint. Save
% those as <recording_id>_actsort_LABELS.mat in output_dir.

opts = parse_options(varargin{:});

neu_h5_path = char(neu_h5_path);
output_dir = char(output_dir);
recording_id = char(recording_id);

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

dataset_path = char(opts.DatasetPath);
preprocess_input = [neu_h5_path ':' dataset_path];
final_h5_path = default_final_h5_path(neu_h5_path);

raw_output_path = fullfile(output_dir, [recording_id '_extract_output_unsorted.mat']);

fprintf('\n=== EXTRACT %s ===\n', recording_id);
fprintf('Input H5:      %s\n', neu_h5_path);
fprintf('Final H5:      %s\n', final_h5_path);
fprintf('Output MAT:    %s\n', raw_output_path);

configure_gpu_forward_compatibility(opts.UseGpu, opts.GpuForwardCompatibility);

if opts.Overwrite || ~exist(final_h5_path, 'file')
    fprintf('Preprocessing %s\n', preprocess_input);
    preprocess_config = get_defaults([]);
    preprocess_config.use_gpu = opts.UseGpu;
    preprocess_save(preprocess_input, preprocess_config);
else
    fprintf('Using existing preprocessed H5: %s\n', final_h5_path);
end

if ~exist(final_h5_path, 'file')
    error('Expected preprocessed H5 was not found: %s', final_h5_path);
end

M = {final_h5_path, dataset_path};

config = [];
config = get_defaults(config);

config.avg_cell_radius = opts.AvgCellRadius;
config.num_partitions_x = opts.NumPartitionsX;
config.num_partitions_y = opts.NumPartitionsY;
config.downsample_time_by = opts.DownsampleTimeBy;
config.use_sparse_arrays = opts.UseSparseArrays;
config.max_iter = opts.MaxIter;
config.thresholds.eccent_thresh = opts.EccentThresh;
config.thresholds.spatial_corrupt_thresh = opts.SpatialCorruptThresh;
config.thresholds.T_dup_corr_thresh = opts.TDupCorrThresh;
config.adaptive_kappa = opts.AdaptiveKappa;

config.preprocess = 0;
config.F_per_pixel = h5read(final_h5_path, '/F_per_pixel');

config.use_gpu = opts.UseGpu;
config.arbitrary_mask = opts.ArbitraryMask;
config.visualize_cellfinding = opts.VisualizeCellfinding;

config.thresholds.T_min_snr = opts.TMinSnr;
config.thresholds.size_upper_limit = opts.SizeUpperLimit;
config.thresholds.size_lower_limit = opts.SizeLowerLimit;

extractOutput = extractor(M, config);
output = extractOutput; %#ok<NASGU>

save(raw_output_path, 'output', '-v7.3');

if opts.MakePlots
    plot_extract_quicklook(extractOutput);
end

summary = struct();
summary.recording_id = recording_id;
summary.neu_h5_path = neu_h5_path;
summary.final_h5_path = final_h5_path;
summary.raw_output_path = raw_output_path;
summary.status = 'completed';
end


function opts = parse_options(varargin)
p = inputParser;
p.addParameter('DatasetPath', '/data');
p.addParameter('UseGpu', 1);
p.addParameter('GpuForwardCompatibility', 1);
p.addParameter('Overwrite', false);
p.addParameter('AvgCellRadius', 5);
p.addParameter('NumPartitionsX', 3);
p.addParameter('NumPartitionsY', 3);
p.addParameter('DownsampleTimeBy', 5);
p.addParameter('UseSparseArrays', 1);
p.addParameter('MaxIter', 5);
p.addParameter('EccentThresh', 5);
p.addParameter('SpatialCorruptThresh', 3.5);
p.addParameter('TDupCorrThresh', 0.7);
p.addParameter('AdaptiveKappa', 2);
p.addParameter('ArbitraryMask', 0);
p.addParameter('VisualizeCellfinding', 1);
p.addParameter('TMinSnr', 6);
p.addParameter('SizeUpperLimit', 3);
p.addParameter('SizeLowerLimit', 0.3);
p.addParameter('MakePlots', false);
p.parse(varargin{:});
opts = p.Results;

opts.UseGpu = numeric_flag(opts.UseGpu);
opts.GpuForwardCompatibility = logical_flag(opts.GpuForwardCompatibility);
opts.Overwrite = logical_flag(opts.Overwrite);
opts.UseSparseArrays = numeric_flag(opts.UseSparseArrays);
opts.ArbitraryMask = numeric_flag(opts.ArbitraryMask);
opts.VisualizeCellfinding = numeric_flag(opts.VisualizeCellfinding);
opts.MakePlots = logical_flag(opts.MakePlots);

opts.AvgCellRadius = numeric_value(opts.AvgCellRadius);
opts.NumPartitionsX = numeric_value(opts.NumPartitionsX);
opts.NumPartitionsY = numeric_value(opts.NumPartitionsY);
opts.DownsampleTimeBy = numeric_value(opts.DownsampleTimeBy);
opts.MaxIter = numeric_value(opts.MaxIter);
opts.EccentThresh = numeric_value(opts.EccentThresh);
opts.SpatialCorruptThresh = numeric_value(opts.SpatialCorruptThresh);
opts.TDupCorrThresh = numeric_value(opts.TDupCorrThresh);
opts.AdaptiveKappa = numeric_value(opts.AdaptiveKappa);
opts.TMinSnr = numeric_value(opts.TMinSnr);
opts.SizeUpperLimit = numeric_value(opts.SizeUpperLimit);
opts.SizeLowerLimit = numeric_value(opts.SizeLowerLimit);
end


function configure_gpu_forward_compatibility(use_gpu, enable_forward_compatibility)
if ~logical(use_gpu)
    return
end

try
    parallel.gpu.enableCUDAForwardCompatibility(enable_forward_compatibility);
    tf = parallel.gpu.enableCUDAForwardCompatibility;
    fprintf('GPU CUDA forward compatibility: %d\n', tf);
catch ME
    warning( ...
        'run_extract_one_record:GpuForwardCompatibility', ...
        'Could not set GPU CUDA forward compatibility: %s', ...
        ME.message);
end
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


function final_h5_path = default_final_h5_path(neu_h5_path)
[folder, name, ext] = fileparts(neu_h5_path);
final_h5_path = fullfile(folder, [name '_final' ext]);
end


function plot_extract_quicklook(output)
plot_output_cellmap(output, 0);

if isfield(output, 'spatial_weights') && isfield(output, 'temporal_weights')
    S_ex = full(output.spatial_weights);
    [~, ~, n_cells] = size(S_ex);
    pick_neurons = 1:min(20, n_cells);
    T_ex = output.temporal_weights';
    max_im = output.info.summary_image;

    plot_simulated_cellmap( ...
        S_ex, max_im, S_ex(:, :, pick_neurons), [1, 0.5, 0], [0, 0.5, 1]);
    plot_stacked_traces_double(T_ex(pick_neurons, :), [], 1, {[0, 0.5, 1]});
end
end
