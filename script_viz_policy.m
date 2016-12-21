% script to visualize trajectories from a trained NN policy

tmp = py.vin.vin; clear tmp;  % to load Python

% set parameters and load NN
size_1 = 28; size_2 = 28;
k = 36;
prior = 'reward';
model = 'vin';
if strcmp(model,'cnn')
    nn = py.cnn.cnn(pyargs('im_size',int32([size_1,size_2]),'batchsize',int32(1),'statebatchsize',int32(1)));
elseif strcmp(model,'vin')
    nn = py.vin.vin(pyargs('im_size',int32([size_1,size_2]),'k',int32(k),'batchsize',int32(1),'statebatchsize',int32(1)));
elseif strcmp(model,'fcn')
    nn = py.FCN.fcn(pyargs('im_size',int32([size_1,size_2]),'batchsize',int32(1),'statebatchsize',int32(1)));    
end
weight_file = './results/grid28_VIN.pk';
nn.load_weights(pyargs('infile',weight_file));

%% Evaluate NN
% Predict trajectories in closed-loop, and compare with shortest path
dom_size = [size_1,size_2];      % domain size
Ndomains = 100;                  % number of domains to evaluate
maxObs = 50;                     % maximum number of obstacles in a domain
maxObsSize = 2.0;                % maximum obstacle size
Ntrajs = 1;                      % trajectories from each domain
numActions = 8;
action_vecs = ([[-1,0; 1,0; 0,1; 0,-1]; 1/sqrt(2)*[-1,1; -1,-1; 1,1; 1,-1]])';  % state difference unit vectors for each action
action_vecs_unnorm = ([-1,0; 1,0; 0,1; 0,-1; -1,1; -1,-1; 1,1; 1,-1]);          % un-normalized state difference vectors
plot_value = false;

% containers for data
numSamples = 1;
numTrajs = 1;
figure(1);
for dom = 1:Ndomains
    % generate random domain
    goal(1,1) = 1+randi(size_1-1);
    goal(1,2) = 1+randi(size_2-1);
    % generate random obstacles
    obs = obstacle_gen(dom_size,goal,maxObsSize);
    n_obs = obs.add_N_rand_obs(randi(maxObs));
    add_border_res = obs.add_border;
    if n_obs == 0 || add_border_res 
        disp('no obstacles added, or problem with border, regenerating map')
        continue;       % no obstacles added, or problem with border, skip
    end
    im = double(rgb2gray(obs.getimage));
    im = max(max(im)) - im; im = im./max(max(im)); imagesc(im); drawnow;
    % make graph (deterministic MDP)
    G = Gridworld_Graph8(im,goal(1),goal(2));
    value_prior = G.getRewardPrior;
    % sample shortest-path trajectories in graph
    [states_xy, states_one_hot] = SampleGraphTraj(G,Ntrajs);
    figure(1); hold on;
    for i = 1:Ntrajs
        if ~isempty(states_xy{i}) && size(states_xy{i},1)>1
            L = size(states_xy{i},1)*2;
            pred_traj = zeros(L,2);
            pred_traj(1,:) = states_xy{i}(1,:);
            for j = 2:L
                % creat state vector and image vector, and save to file
                state_xy_data = uint8([pred_traj(j-1,1)-1, pred_traj(j-1,2)-1]);
                im_data = uint8(reshape(im',1,[]));
                value_data = uint8(reshape(value_prior',1,[]));
                % call NN to predict action from input file (passing data directly from Matlab to python is difficult)
                save('test_input.mat','im_data','value_data','state_xy_data');
                a = nn.predict(pyargs('input', 'test_input.mat'))+1;
                % calculate next state based on action
                s = G.map_ind_to_state(pred_traj(j-1,1),pred_traj(j-1,2));
                ns = G.sampleNextState(s,a);
                [nr,nc] = G.getCoords(ns);
                pred_traj(j,2) = nc;
                pred_traj(j,1) = nr;
                if (nr == goal(1)) && (nc == goal(2))
                    pred_traj(j+1:end,2) = nc;
                    pred_traj(j+1:end,1) = nr;
                    break;
                end
            end
            % plot stuff
			figure(1);
            plot(states_xy{i}(:,2),states_xy{i}(:,1));drawnow;
            plot(pred_traj(:,2),pred_traj(:,1),'-X');drawnow;
            legend('Shortest path','Predicted path');
            plot(states_xy{i}(1,2),states_xy{i}(1,1),'-o');drawnow;
            plot(states_xy{i}(end,2),states_xy{i}(end,1),'-s');drawnow;
            hold off;
            if plot_value
                figure(2);
                pred_val = nn.predict_value(pyargs('input', 'test_input.mat'));
                val_map = python_ndarray_to_matrix(pred_val(1),[size_1,size_2]);
                r_map = python_ndarray_to_matrix(pred_val(2),[size_1,size_2]);
                subplot(1,2,1);
                imagesc(r_map);
                title('Learned Reward');
                subplot(1,2,2);
                imagesc(val_map);
                title('Learned Value');
                drawnow;
            end
            pause;%(0.6);
        end
    end
end
