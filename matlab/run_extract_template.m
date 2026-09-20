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
%   extract_overwrite=0
%   extract_visualize_cellfinding=1
%   extract_t_min_snr=6
%   extract_make_plots=0
%
% Example:
%   python scripts/run_matlab_neural_extraction.py preprocess_out/manifest_with_h5.csv ...

assert(exist('recording_id', 'var') == 1, 'Missing recording_id');
assert(exist('session_id', 'var') == 1, 'Missing session_id');
assert(exist('neu_h5_path', 'var') == 1, 'Missing neu_h5_path');
assert(exist('output_dir', 'var') == 1, 'Missing output_dir');

template_dir = fileparts(mfilename('fullpath'));
addpath(template_dir);

dataset_path = get_optional_value('dataset_path', '/data');
extract_use_gpu = get_optional_value('extract_use_gpu', 1);
extract_overwrite = get_optional_value('extract_overwrite', false);
extract_visualize_cellfinding = get_optional_value('extract_visualize_cellfinding', 1);
extract_t_min_snr = get_optional_value('extract_t_min_snr', 6);
extract_make_plots = get_optional_value('extract_make_plots', false);

fprintf('Recording: %s\n', recording_id);
fprintf('Session:   %s\n', session_id);
fprintf('Input H5:  %s\n', neu_h5_path);
fprintf('Output:    %s\n', output_dir);

summary = run_extract_one_record( ...
    neu_h5_path, ...
    output_dir, ...
    recording_id, ...
    'DatasetPath', dataset_path, ...
    'UseGpu', extract_use_gpu, ...
    'Overwrite', extract_overwrite, ...
    'VisualizeCellfinding', extract_visualize_cellfinding, ...
    'TMinSnr', extract_t_min_snr, ...
    'MakePlots', extract_make_plots);

disp(summary);

fprintf('\nManual curation checkpoint:\n');
fprintf('Open ActSort/manualActSort for this recording, classify accepted cells, and save labels to:\n');
fprintf('  %s\n', fullfile(output_dir, [recording_id '_precomputed_output_LABELS.mat']));


function value = get_optional_value(name, default_value)
if evalin('base', sprintf('exist(''%s'', ''var'')', name))
    value = evalin('base', name);
else
    value = default_value;
end
end
