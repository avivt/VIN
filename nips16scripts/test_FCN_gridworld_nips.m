% script to test FCN networks in NIPS experiments
data_dir = './data/nips16/';
% %% Test on 8x8 map 
test_file = [data_dir 'gridworld_8_test.mat']; 
weight_file = './nips16results/gridworld/grid8_FCN.pk';
model = 'FCN';
k = 0;
size_1 = 8;
size_2 = 8;
[optimal_lengths,pred_lengths] = test_network(model, weight_file, test_file, [size_1,size_2], k);
ind_no_obs = find(pred_lengths > 0);
mean_succ_8 = mean(pred_lengths > 0);
mean_traj_diff_8 = mean(pred_lengths(ind_no_obs) - optimal_lengths(ind_no_obs));

%% Test on 16x16 map 
test_file = [data_dir 'gridworld_16_test.mat']; 
weight_file = './nips16results/gridworld/grid16_FCN.pk';
model = 'FCN';
k = 0;
size_1 = 16;
size_2 = 16;
[optimal_lengths,pred_lengths] = test_network(model, weight_file, test_file, [size_1,size_2], k);
ind_no_obs = find(pred_lengths > 0);
mean_succ_16 = mean(pred_lengths > 0);
mean_traj_diff_16 = mean(pred_lengths(ind_no_obs) - optimal_lengths(ind_no_obs));

%% Test on 28x28 map 
test_file = [data_dir 'gridworld_28_test.mat']; 
weight_file = './nips16results/gridworld/grid28_FCN.pk';
model = 'FCN';
k = 0;
size_1 = 28;
size_2 = 28;
[optimal_lengths,pred_lengths] = test_network(model, weight_file, test_file, [size_1,size_2], k);
ind_no_obs = find(pred_lengths > 0);
mean_succ_28 = mean(pred_lengths > 0);
mean_traj_diff_28 = mean(pred_lengths(ind_no_obs) - optimal_lengths(ind_no_obs));
