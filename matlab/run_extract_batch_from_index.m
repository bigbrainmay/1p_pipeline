function batch_results = run_extract_batch_from_index(manifest_csv, output_root, varargin)
%RUN_EXTRACT_BATCH_FROM_INDEX Loop EXTRACT over pipeline H5 rows in MATLAB.
%
% Default input:
%   run_extract_batch_from_index
%
% Typical GPU-PC input:
%   run_extract_batch_from_index('preprocess_out/manifest_with_h5.csv', 'preprocess_out')
%
% One recording first:
%   run_extract_batch_from_index( ...
%       'preprocess_out/manifest_with_h5.csv', ...
%       'preprocess_out', ...
%       'Only', 'C57-2-1-B_20260807_beh-cam_1')

if nargin < 1 || isempty(manifest_csv)
    manifest_csv = fullfile('preprocess_out', 'manifest_with_h5.csv');
end
if nargin < 2 || isempty(output_root)
    output_root = 'preprocess_out';
end

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

opts = parse_batch_options(varargin{:});

manifest_csv = char(manifest_csv);
output_root = char(output_root);

T = readtable(manifest_csv, 'PreserveVariableNames', true);
required_cols = {'recording_id', 'neu_h5'};
for i = 1:numel(required_cols)
    assert(any(strcmp(T.Properties.VariableNames, required_cols{i})), ...
        'Manifest is missing required column: %s', required_cols{i});
end

rows = {};
for i = 1:height(T)
    recording_id = table_text(T, i, 'recording_id');
    session_id = table_text(T, i, 'session_id');
    trial_type = table_text(T, i, 'trial_type');
    neu_h5_path = table_text(T, i, 'neu_h5');

    if ~selected_recording(opts.Only, recording_id, session_id, trial_type)
        continue
    end

    if missing_text(neu_h5_path)
        rows(end + 1, :) = {recording_id, session_id, neu_h5_path, '', 'missing_neu_h5', ''}; %#ok<AGROW>
        continue
    end

    output_dir = table_text(T, i, 'matlab_output_dir');
    if missing_text(output_dir)
        output_dir = fullfile(output_root, recording_id, 'matlab');
    end

    try
        summary = run_extract_one_record( ...
            neu_h5_path, ...
            output_dir, ...
            recording_id, ...
            'DatasetPath', opts.DatasetPath, ...
            'UseGpu', opts.UseGpu, ...
            'GpuForwardCompatibility', opts.GpuForwardCompatibility, ...
            'Overwrite', opts.Overwrite, ...
            'VisualizeCellfinding', opts.VisualizeCellfinding, ...
            'TMinSnr', opts.TMinSnr, ...
            'MakePlots', opts.MakePlots);

        rows(end + 1, :) = { ...
            recording_id, ...
            session_id, ...
            neu_h5_path, ...
            output_dir, ...
            summary.status, ...
            ''}; %#ok<AGROW>
    catch ME
        rows(end + 1, :) = { ...
            recording_id, ...
            session_id, ...
            neu_h5_path, ...
            output_dir, ...
            'error', ...
            ME.message}; %#ok<AGROW>
        fprintf(2, '%s: ERROR %s\n', recording_id, ME.message);
        if opts.FailFast
            rethrow(ME);
        end
    end
end

if isempty(rows)
    rows = cell(0, 6);
end

batch_results = cell2table(rows, 'VariableNames', { ...
    'recording_id', ...
    'session_id', ...
    'neu_h5', ...
    'matlab_output_dir', ...
    'status', ...
    'error'});

if ~exist(output_root, 'dir')
    mkdir(output_root);
end
index_path = fullfile(output_root, 'matlab_extraction_index_from_matlab.csv');
writetable(batch_results, index_path);
fprintf('Wrote %s\n', index_path);
end


function opts = parse_batch_options(varargin)
p = inputParser;
p.addParameter('Only', '');
p.addParameter('DatasetPath', '/data');
p.addParameter('UseGpu', 1);
p.addParameter('GpuForwardCompatibility', 1);
p.addParameter('Overwrite', false);
p.addParameter('VisualizeCellfinding', 1);
p.addParameter('TMinSnr', 6);
p.addParameter('MakePlots', false);
p.addParameter('FailFast', false);
p.parse(varargin{:});
opts = p.Results;
end


function value = table_text(T, row_idx, col_name)
if ~any(strcmp(T.Properties.VariableNames, col_name))
    value = '';
    return
end

raw = T.(col_name)(row_idx);
if iscell(raw)
    raw = raw{1};
end
value = char(string(raw));
end


function tf = missing_text(value)
value = strtrim(char(value));
tf = isempty(value) || any(strcmpi(value, {'<missing>', 'missing', 'nan', 'none', 'null'}));
end


function tf = selected_recording(only, recording_id, session_id, trial_type)
only = strtrim(char(only));
if isempty(only)
    tf = true;
    return
end
tf = any(strcmp(only, {recording_id, session_id, trial_type}));
end
