function [mask, rect] = select_motion_correction_mask(input_h5_path, output_mat_path, varargin)
%SELECT_MOTION_CORRECTION_MASK Draw and save a NormCorre motion-correction mask.
%
% Example:
%   select_motion_correction_mask( ...
%       'D:\preprocess_out\C5731A_20260806\neural\C5731A_20260806_miniscope.h5', ...
%       'D:\preprocess_out\C5731A_20260806\neural\C5731A_20260806_mc_mask.mat');
%
% The saved MAT file contains:
%   mask: binary [height x width] mask
%   rect: [x y width height] rectangle in image coordinates

opts = parse_options(varargin{:});
input_h5_path = char(input_h5_path);
output_mat_path = char(output_mat_path);
dataset_path = char(opts.DatasetPath);

info = h5info(input_h5_path, dataset_path);
dims = info.Dataspace.Size;
height = dims(1);
width = dims(2);
n_frames = min(opts.NumFrames, dims(3));

fprintf('Reading %d frames from %s:%s\n', n_frames, input_h5_path, dataset_path);
movie = single(h5read(input_h5_path, dataset_path, [1, 1, 1], [height, width, n_frames]));
projection = mean(movie, 3);
clear movie

figure('Name', 'Motion correction mask');
imshow(projection, []);
axis image
title({'Draw the motion-correction mask', 'Double-click inside the rectangle when done'});

if exist('drawrectangle', 'file') == 2
    roi = drawrectangle('Color', 'y');
    wait(roi);
    rect = roi.Position;
else
    rect = getrect;
end

rect = round(rect);
x_min = max(1, rect(1));
y_min = max(1, rect(2));
x_max = min(width, rect(1) + rect(3));
y_max = min(height, rect(2) + rect(4));
if x_max < x_min || y_max < y_min
    error('Selected rectangle is outside the movie bounds: [%s]', num2str(rect));
end

mask = zeros(height, width);
mask(y_min:y_max, x_min:x_max) = 1;

output_parent = fileparts(output_mat_path);
if ~isempty(output_parent) && ~exist(output_parent, 'dir')
    mkdir(output_parent);
end

save(output_mat_path, 'mask', 'rect', 'input_h5_path', 'dataset_path', 'height', 'width');
fprintf('Saved motion-correction mask: %s\n', output_mat_path);
end


function opts = parse_options(varargin)
p = inputParser;
p.addParameter('DatasetPath', '/data');
p.addParameter('NumFrames', 100);
p.parse(varargin{:});
opts = p.Results;

if ischar(opts.NumFrames) || isstring(opts.NumFrames)
    opts.NumFrames = str2double(opts.NumFrames);
end
end
