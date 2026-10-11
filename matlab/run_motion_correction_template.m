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
%   mc_get_mask=1
%
% Example:
%   python scripts/run_matlab_motion_correction.py preprocess_out/manifest_with_h5.csv ...

assert(exist('recording_id', 'var') == 1, 'Missing recording_id');
assert(exist('session_id', 'var') == 1, 'Missing session_id');
assert(exist('neu_h5_path', 'var') == 1, 'Missing neu_h5_path');
assert(exist('neu_mc_h5_path', 'var') == 1, 'Missing neu_mc_h5_path');

template_dir = fileparts(mfilename('fullpath'));
addpath(template_dir);

dataset_path = get_optional_value('dataset_path', '/data');
mc_overwrite = get_optional_value('mc_overwrite', false);
mc_prefilter_enable = get_optional_value('mc_prefilter_enable', true);
mc_prefilter_chunk_size = get_optional_value('mc_prefilter_chunk_size', 1000);
mc_prefilter_avg_cell_radius = get_optional_value('mc_prefilter_avg_cell_radius', 6);
mc_prefilter_spatial_highpass_cutoff = get_optional_value('mc_prefilter_spatial_highpass_cutoff', 5);
mc_prefilter_spatial_lowpass_cutoff = get_optional_value('mc_prefilter_spatial_lowpass_cutoff', 2);
mc_prefilter_use_gpu = get_optional_value('mc_prefilter_use_gpu', 0);
mc_keep_prefilter = get_optional_value('mc_keep_prefilter', false);
mc_get_mask = get_optional_value('mc_get_mask', 1);

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
    'GetMask', mc_get_mask);

disp(summary);


function value = get_optional_value(name, default_value)
if evalin('base', sprintf('exist(''%s'', ''var'')', name))
    value = evalin('base', name);
else
    value = default_value;
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
