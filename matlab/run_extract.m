function verify = run_extract(prep_path, final_h5_dst, unsort_mat_dst)

config = get_defaults([]);
config.use_gpu = 1;

preprocess_save(strcat(prep_path,':\data'),config); % ":/data" accesses that field of the file
M = h5read(final_h5_dst,'/data');

config =[];
config = get_defaults(config);

config.avg_cell_radius=5;
config.num_partitions_x=1;
config.num_partitions_y=1;
config.downsample_time_by = 5;
config.use_sparse_arrays = 1;
config.max_iter = 5;
config.threshholds.eccent_thresh = 5;
config.thresholds.spatial_corrupt_thresh = 3.5;
config.thresholds.T_dup_corr_thresh = 0.7;
config.adaptive_kappa = 2;

config.preprocess = 0;
config.F_per_pixel = h5read(final_h5_dst,'/F_per_pixel');

config.use_gpu = 1;
config.arbitrary_mask = 0;
config.visualize_cellfinding = 1;

% change these as needed
config.thresholds.T_min_snr=6; % can go as low as 3.3

config.thresholds.size_upper_limit = 3;
config.thresholds.size_lower_limit = 0.3;

output=extractor(M,config);
save(unsort_mat_dst,'output','-v7.3');

verify = 1;

%% A more advanced plotting

S_ex = full(output.spatial_weights);
[nx,ny,~] = size(M);
S_ex = reshape(S_ex,nx,ny,[]);
T_ex = output.temporal_weights';

pick_ner = [1:20];
max_im = output.info.summary_image ;

plot_simulated_cellmap(S_ex, ...
    max_im,S_ex(:,:,pick_ner),[1,0.5,0],[0,0.5,1])

plot_stacked_traces_double(T_ex(pick_ner,:), [],1,{[0,0.5,1]});