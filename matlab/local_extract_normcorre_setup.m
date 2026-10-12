% Local MATLAB path setup for EXTRACT + NoRMCorre batch runs.
%
% Pass this file to the Python runners with:
%   --extra "matlab_setup_path=C:\Users\ONIX\1p_pipeline\matlab\local_extract_normcorre_setup.m"
%
% This file is intentionally small and local-machine oriented. It adds the
% usual MATLAB Add-Ons EXTRACT location and a user-level NoRMCorre checkout.

repo_matlab_dir = fileparts(mfilename('fullpath'));
addpath(repo_matlab_dir);

user_profile = getenv('USERPROFILE');
app_data = getenv('APPDATA');

candidate_roots = {
    fullfile(app_data, 'MathWorks', 'MATLAB Add-Ons', 'Collections', 'EXTRACT-public'), ...
    fullfile(user_profile, 'MATLAB', 'NoRMCorre'), ...
    fullfile(user_profile, 'Documents', 'MATLAB', 'NoRMCorre')
};

for i = 1:numel(candidate_roots)
    add_path_if_present(candidate_roots{i});
end

fprintf('EXTRACT setup check:\n');
fprintf('  get_defaults:        %s\n', which_or_missing('get_defaults'));
fprintf('  run_normcorre:       %s\n', which_or_missing('run_normcorre_pipeline'));
fprintf('  NoRMCorreSetParms:   %s\n', which_or_missing('NoRMCorreSetParms'));
fprintf('  normcorre_batch:     %s\n', which_or_missing('normcorre_batch'));

if isempty(which('NoRMCorreSetParms'))
    error(['NoRMCorreSetParms is still missing. Install NoRMCorre into ', ...
        fullfile(user_profile, 'MATLAB', 'NoRMCorre'), ...
        ' or edit matlab/local_extract_normcorre_setup.m to add its folder.']);
end


function add_path_if_present(path_value)
if isempty(path_value)
    return
end
if exist(path_value, 'dir') == 7
    fprintf('Adding MATLAB path recursively: %s\n', path_value);
    addpath(genpath(path_value));
else
    fprintf('MATLAB path not found, skipping: %s\n', path_value);
end
end


function value = which_or_missing(function_name)
value = which(function_name);
if isempty(value)
    value = '<missing>';
end
end
