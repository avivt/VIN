% script to test VIN networks with partial training data in NIPS experiments
data_dir = './data/nips16/';
%% Test on VIN on 16x16 map 20% data
test_file = [data_dir 'gridworld_16_test.mat']; 
weight_file = './nips16results/gridworld/grid16_VIN_02_data.pk';
model = 'VIN';
k = 20;
size_1 = 16;
size_2 = 16;
[optimal_lengths,pred_lengths] = test_network(model, weight_file, test_file, [size_1,size_2], k);
ind_no_obs = find(pred_lengths > 0);
mean_succ_16_20 = mean(pred_lengths > 0);
mean_traj_diff_16_20 = mean(pred_lengths(ind_no_obs) - optimal_lengths(ind_no_obs));

%% Test on VIN on 16x16 map 50% data
test_file = [data_dir 'gridworld_16_test.mat']; 
weight_file = './nips16results/gridworld/grid16_VIN_05_data.pk';
model = 'VIN';
k = 20;
size_1 = 16;
size_2 = 16;
[optimal_lengths,pred_lengths] = test_network(model, weight_file, test_file, [size_1,size_2], k);
ind_no_obs = find(pred_lengths > 0);
mean_succ_16_50 = mean(pred_lengths > 0);
mean_traj_diff_16_50 = mean(pred_lengths(ind_no_obs) - optimal_lengths(ind_no_obs));

%% Test on VIN on 16x16 map 100% data
test_file = [data_dir 'gridworld_16_test.mat']; 
weight_file = './icml16results/gridworld/grid16_VIN.pk';
model = 'VIN';
k = 20;
size_1 = 16;
size_2 = 16;
[optimal_lengths,pred_lengths] = test_network(model, weight_file, test_file, [size_1,size_2], k);
ind_no_obs = find(pred_lengths > 0);
mean_succ_16_100 = mean(pred_lengths > 0);
mean_traj_diff_16_100 = mean(pred_lengths(ind_no_obs) - optimal_lengths(ind_no_obs));

%% Test on untied VIN on 16x16 map 20% data
test_file = [data_dir 'gridworld_16_test.mat']; 
weight_file = './nips16results/gridworld/grid16_VIN_untied_02_data.pk';
model = 'untiedVIN';
k = 20;
size_1 = 16;
size_2 = 16;
[optimal_lengths,pred_lengths] = test_network(model, weight_file, test_file, [size_1,size_2], k);
ind_no_obs = find(pred_lengths > 0);
mean_succ_untied_16_20 = mean(pred_lengths > 0);
mean_traj_diff_untied_16_20 = mean(pred_lengths(ind_no_obs) - optimal_lengths(ind_no_obs));

%% Test on untied VIN on 16x16 map 50% data
test_file = [data_dir 'gridworld_16_test.mat']; 
weight_file = './nips16results/gridworld/grid16_VIN_untied_05_data.pk';
model = 'untiedVIN';
k = 20;
size_1 = 16;
size_2 = 16;
[optimal_lengths,pred_lengths] = test_network(model, weight_file, test_file, [size_1,size_2], k);
ind_no_obs = find(pred_lengths > 0);
mean_succ_untied_16_50 = mean(pred_lengths > 0);
mean_traj_diff_untied_16_50 = mean(pred_lengths(ind_no_obs) - optimal_lengths(ind_no_obs));

%% Test on untied VIN on 16x16 map 100% data
test_file = [data_dir 'gridworld_16_test.mat']; 
weight_file = './nips16results/gridworld/grid16_VIN_untied.pk';
model = 'untiedVIN';
k = 20;
size_1 = 16;
size_2 = 16;
[optimal_lengths,pred_lengths] = test_network(model, weight_file, test_file, [size_1,size_2], k);
ind_no_obs = find(pred_lengths > 0);
mean_succ_untied_16_100 = mean(pred_lengths > 0);
mean_traj_diff_untied_16_100 = mean(pred_lengths(ind_no_obs) - optimal_lengths(ind_no_obs));