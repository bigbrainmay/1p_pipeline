% Called by scripts/run_matlab_neural_extraction.py.
%
% The Python runner defines these variables before this file is run:
%   recording_id
%   session_id
%   neu_h5_path
%   output_dir
%
% Optional variables can be passed from Python with --extra NAME=VALUE:
%   dataset_path=/data
%   extract_use_gpu=1
%   extract_gpu_forward_compatibility=1
%   extract_overwrite=0
%   extract_visualize_cellfinding=1
%   extract_avg_cell_radius=5
%   extract_preprocess_spatial_highpass_cutoff=5
%   extract_preprocess_spatial_lowpass_cutoff=inf
%   extract_preprocess_partition_size_time=500
%   extract_cellfind_spatial_lowpass_cutoff=inf
%   extract_downsample_time_by=5
%   extract_max_iter=8
%   extract_spatial_corrupt_thresh=2.5
%   extract_t_dup_corr_thresh=0.9
%   extract_eccent_thresh=5
%   extract_t_min_snr=6
%   extract_size_upper_limit=2
%   extract_size_lower_limit=0.3
%   extract_make_plots=0
%   matlab_setup_path=''  % EXTRACT_setup.m file or folder to add recursively
%
% Example:
%   python scripts/run_matlab_neural_extraction.py preprocess_out/manifest_with_h5.csv ...

assert(exist('recording_id', 'var') == 1, 'Missing recording_id');
assert(exist('session_id', 'var') == 1, 'Missing session_id');
assert(exist('neu_h5_path', 'var') == 1, 'Missing neu_h5_path');
assert(exist('output_dir', 'var') == 1, 'Missing output_dir');

template_dir = fileparts(mfilename('fullpath'));
addpath(template_dir);

matlab_setup_path = get_optional_value('matlab_setup_path', '');
run_optional_setup(matlab_setup_path);

dataset_path = get_optional_value('dataset_path', '/data');
extract_use_gpu = get_optional_value('extract_use_gpu', 1);
extract_gpu_forward_compatibility = get_optional_value('extract_gpu_forward_compatibility', extract_use_gpu);
extract_overwrite = get_optional_value('extract_overwrite', false);
extract_visualize_cellfinding = get_optional_value('extract_visualize_cellfinding', 1);
extract_avg_cell_radius = get_optional_value('extract_avg_cell_radius', 5);
extract_preprocess_spatial_highpass_cutoff = get_optional_value('extract_preprocess_spatial_highpass_cutoff', 5);
extract_preprocess_spatial_lowpass_cutoff = get_optional_value('extract_preprocess_spatial_lowpass_cutoff', inf);
extract_preprocess_partition_size_time = get_optional_value('extract_preprocess_partition_size_time', 500);
extract_cellfind_spatial_lowpass_cutoff = get_optional_value('extract_cellfind_spatial_lowpass_cutoff', inf);
extract_downsample_time_by = get_optional_value('extract_downsample_time_by', 5);
extract_max_iter = get_optional_value('extract_max_iter', 8);
extract_spatial_corrupt_thresh = get_optional_value('extract_spatial_corrupt_thresh', 2.5);
extract_t_dup_corr_thresh = get_optional_value('extract_t_dup_corr_thresh', 0.9);
extract_eccent_thresh = get_optional_value('extract_eccent_thresh', 5);
extract_t_min_snr = get_optional_value('extract_t_min_snr', 5);
extract_size_upper_limit = get_optional_value('extract_size_upper_limit', 2);
extract_size_lower_limit = get_optional_value('extract_size_lower_limit', 0.3);
extract_make_plots = get_optional_value('extract_make_plots', false);

fprintf('Recording: %s\n', recording_id);
fprintf('Session:   %s\n', session_id);
fprintf('Input H5:  %s\n', neu_h5_path);
fprintf('Output:    %s\n', output_dir);

extract_args = { ...
    'DatasetPath', dataset_path, ...
    'UseGpu', extract_use_gpu, ...
    'GpuForwardCompatibility', extract_gpu_forward_compatibility, ...
    'Overwrite', extract_overwrite, ...
    'VisualizeCellfinding', extract_visualize_cellfinding, ...
    'AvgCellRadius', extract_avg_cell_radius, ...
    'PreprocessSpatialHighpassCutoff', extract_preprocess_spatial_highpass_cutoff, ...
    'PreprocessSpatialLowpassCutoff', extract_preprocess_spatial_lowpass_cutoff, ...
    'PreprocessPartitionSizeTime', extract_preprocess_partition_size_time, ...
    'CellfindSpatialLowpassCutoff', extract_cellfind_spatial_lowpass_cutoff, ...
    'DownsampleTimeBy', extract_downsample_time_by, ...
    'MaxIter', extract_max_iter, ...
    'SpatialCorruptThresh', extract_spatial_corrupt_thresh, ...
    'TDupCorrThresh', extract_t_dup_corr_thresh, ...
    'EccentThresh', extract_eccent_thresh, ...
    'TMinSnr', extract_t_min_snr, ...
    'SizeUpperLimit', extract_size_upper_limit, ...
    'SizeLowerLimit', extract_size_lower_limit, ...
    'MakePlots', extract_make_plots};

summary = run_extract_one_record( ...
    neu_h5_path, ...
    output_dir, ...
    recording_id, ...
    extract_args{:});

disp(summary);

fprintf('\nManual curation checkpoint:\n');
fprintf('Use ActSort/manualActSort on the unsorted EXTRACT output, classify accepted cells, and save labels to:\n');
fprintf('  %s\n', fullfile(output_dir, [recording_id '_precomputed_output_LABELS.mat']));


function value = get_optional_value(name, default_value)
if evalin('base', sprintf('exist(''%s'', ''var'')', name))
    value = evalin('base', name);
else
    value = default_value;
end
end


function run_optional_setup(setup_path)
if missing_text(setup_path)
    return
end

setup_path = char(setup_path);
if exist(setup_path, 'file') == 2
    fprintf('Running MATLAB setup script: %s\n', setup_path);
    addpath(genpath(fileparts(setup_path)));
    run(setup_path);
elseif exist(setup_path, 'dir') == 7
    fprintf('Adding MATLAB setup folder recursively: %s\n', setup_path);
    addpath(genpath(setup_path));
else
    error('matlab_setup_path does not exist: %s', setup_path);
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
