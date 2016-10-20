% script to make data for nips CNN experiments
clear all;
data_dir = '/home/aviv/Data/LearnTraj/nips16/';
dodraw = false;
%% Generate 8x8 map data
data_file = 'gridworld_8_state_channel.mat'; 
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
script_make_data_for_CNN;
clear all;

%% Generate 16x16 map data
data_dir = '/home/aviv/Data/LearnTraj/nips16/';
data_file = 'gridworld_16_state_channel.mat'; 
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
script_make_data_for_CNN;
clear all;

%% Generate 28x28 map data
data_dir = '/home/aviv/Data/LearnTraj/nips16/';
data_file = 'gridworld_28_state_channel.mat'; 
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
script_make_data_for_CNN;
clear all;

%% Generate 36x36 map data
data_dir = '/home/aviv/Data/LearnTraj/nips16/';
data_file = 'gridworld_36_state_channel.mat'; 
size_1 = 36;
size_2 = 36;
add_border = true;
maxObs = 70;
maxObsSize = 3.0;
Ndomains = 5000;
Ntrajs = 7;
prior = 'reward';
rand_goal = true;
zero_min_action = true;
state_batch_size = 1;
script_make_data_for_CNN;
clear all;