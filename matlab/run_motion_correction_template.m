% Called by scripts/run_matlab_motion_correction.py.
%
% The Python runner defines these variables before this file is run:
%   recording_id
%   session_id
%   neu_h5_path
%   neu_mc_h5_path
%   neu_mc_prefilter_h5_path
%
% Optional variables can be passed from Python with --extra NAME=VALUE:
%   dataset_path=/data
%   mc_overwrite=0
%   mc_prefilter_enable=1
%   mc_prefilter_chunk_size=1000
%   mc_prefilter_avg_cell_radius=6
%   mc_prefilter_spatial_highpass_cutoff=5
%   mc_prefilter_spatial_lowpass_cutoff=2
%   mc_prefilter_use_gpu=0
%   mc_keep_prefilter=0
%   mc_get_mask=0
%   mc_mask_rect=[]  % [x y width height], same coordinates returned by getrect
%   mc_mask_mat_path=''  % optional .mat file containing a mask variable
%   mc_bandpass=[]  % defaults to 0 when mc_prefilter_enable=1, else 1
%   mc_nt_template=100
%   mc_num_frame=1000
%   mc_nonrigid=0
%   mc_ns_nonrigid=128
%   mc_avg_cell_radius=7
%   mc_use_gpu=0
%   mc_template=0
%   matlab_setup_path=''  % EXTRACT_setup.m file or folder to add recursively
%
% Example:
%   python scripts/run_matlab_motion_correction.py preprocess_out/manifest_with_h5.csv ...

assert(exist('recording_id', 'var') == 1, 'Missing recording_id');
assert(exist('session_id', 'var') == 1, 'Missing session_id');
assert(exist('neu_h5_path', 'var') == 1, 'Missing neu_h5_path');
assert(exist('neu_mc_h5_path', 'var') == 1, 'Missing neu_mc_h5_path');

template_dir = fileparts(mfilename('fullpath'));
addpath(template_dir);

matlab_setup_path = get_optional_value('matlab_setup_path', '');
run_optional_setup(matlab_setup_path);

dataset_path = get_optional_value('dataset_path', '/data');
mc_overwrite = get_optional_value('mc_overwrite', false);
mc_prefilter_enable = get_optional_value('mc_prefilter_enable', true);
mc_prefilter_chunk_size = get_optional_value('mc_prefilter_chunk_size', 1000);
mc_prefilter_avg_cell_radius = get_optional_value('mc_prefilter_avg_cell_radius', 6);
mc_prefilter_spatial_highpass_cutoff = get_optional_value('mc_prefilter_spatial_highpass_cutoff', inf);
mc_prefilter_spatial_lowpass_cutoff = get_optional_value('mc_prefilter_spatial_lowpass_cutoff', inf);
mc_prefilter_use_gpu = get_optional_value('mc_prefilter_use_gpu', 1);
mc_keep_prefilter = get_optional_value('mc_keep_prefilter', false);
mc_get_mask = get_optional_value('mc_get_mask', 0);
mc_mask_rect = get_optional_value('mc_mask_rect', []);
mc_mask_mat_path = get_optional_value('mc_mask_mat_path', '');
mc_bandpass = get_optional_value('mc_bandpass', []);
mc_nt_template = get_optional_value('mc_nt_template', 100);
mc_num_frame = get_optional_value('mc_num_frame', 1000);
mc_nonrigid = get_optional_value('mc_nonrigid', 0);
mc_ns_nonrigid = get_optional_value('mc_ns_nonrigid', 128);
mc_avg_cell_radius = get_optional_value('mc_avg_cell_radius', 7);
mc_use_gpu = get_optional_value('mc_use_gpu', 0);
mc_template = get_optional_value('mc_template', 0);

if ~exist('neu_mc_prefilter_h5_path', 'var') || missing_text(neu_mc_prefilter_h5_path)
    [folder, name, ext] = fileparts(neu_mc_h5_path);
    neu_mc_prefilter_h5_path = fullfile(folder, [name '_prefilter' ext]);
end

summary = run_motion_correction_one_record( ...
    neu_h5_path, ...
    neu_mc_h5_path, ...
    recording_id, ...
    'DatasetPath', dataset_path, ...
    'Overwrite', mc_overwrite, ...
    'PrefilterEnable', mc_prefilter_enable, ...
    'PrefilterH5Path', neu_mc_prefilter_h5_path, ...
    'PrefilterChunkSize', mc_prefilter_chunk_size, ...
    'PrefilterAvgCellRadius', mc_prefilter_avg_cell_radius, ...
    'PrefilterSpatialHighpassCutoff', mc_prefilter_spatial_highpass_cutoff, ...
    'PrefilterSpatialLowpassCutoff', mc_prefilter_spatial_lowpass_cutoff, ...
    'PrefilterUseGpu', mc_prefilter_use_gpu, ...
    'KeepPrefilter', mc_keep_prefilter, ...
    'GetMask', mc_get_mask, ...
    'MaskRect', mc_mask_rect, ...
    'MaskMatPath', mc_mask_mat_path, ...
    'Bandpass', mc_bandpass, ...
    'NtTemplate', mc_nt_template, ...
    'NumFrame', mc_num_frame, ...
    'Nonrigid', mc_nonrigid, ...
    'NsNonrigid', mc_ns_nonrigid, ...
    'McAvgCellRadius', mc_avg_cell_radius, ...
    'McUseGpu', mc_use_gpu, ...
    'McTemplate', mc_template);

disp(summary);


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
