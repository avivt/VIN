% script to make data for nips CNN experiments
clear all;
data_dir = './data';
dodraw = false;
%% Generate 8x8 map data
data_file = 'gridworld_8.mat'; 
size_1 = 8;
size_2 = 8;
add_border = true;
maxObs = 30;
maxObsSize = 0.0;
Ndomains = 5000;
Ntrajs = 7;
prior = 'reward';
rand_goal = true;
zero_min_action = true;
state_batch_size = 1;
script_make_data;
clear all;

%% Generate 16x16 map data
data_dir = './data';
data_file = 'gridworld_16.mat'; 
size_1 = 16;
size_2 = 16;
add_border = true;
maxObs = 40;
maxObsSize = 1.0;
Ndomains = 5000;
Ntrajs = 7;
prior = 'reward';
rand_goal = true;
zero_min_action = true;
state_batch_size = 1;
script_make_data;
clear all;

%% Generate 28x28 map data
data_dir = './data';
data_file = 'gridworld_28.mat'; 
size_1 = 28;
size_2 = 28;
add_border = true;
maxObs = 50;
maxObsSize = 2.0;
Ndomains = 5000;
Ntrajs = 7;
prior = 'reward';
rand_goal = true;
zero_min_action = true;
state_batch_size = 1;
script_make_data;
clear all;
