function [optimal_lengths,pred_lengths] = test_network(model, weight_file, test_file, imsize, k)
% script to evaluate success rate of network on a test-set of trajectories

tmp = py.convNN.convNN; clear tmp;  % to load Python
size_1 = imsize(1); size_2 = imsize(2);
if strcmp(model,'VIN')
    nn = py.convBatch.convBatch(pyargs('im_size',int32([size_1,size_2]),'k',int32(k),'batchsize',int32(1),'statebatchsize',int32(1)));
elseif strcmp(model,'untiedVIN')
    nn = py.vin_untied.vin_untied(pyargs('im_size',int32([size_1,size_2]),'k',int32(k),'batchsize',int32(1),'statebatchsize',int32(1)));
elseif strcmp(model,'FCN')
    nn = py.FCN.fcn(pyargs('im_size',int32([size_1,size_2]),'batchsize',int32(1),'statebatchsize',int32(1)));
elseif strcmp(model,'CNN')
    nn = py.CNN.cnn(pyargs('im_size',int32([size_1,size_2]),'batchsize',int32(1)));
end
nn.load_weights(pyargs('infile',weight_file));
load(test_file);
%% Evaluate NN
% Predict trajectories in closed-loop, and compare with shortest path
Ndomains = size(all_im_data,1);                  % number of domains

% containers for data
optimal_lengths = zeros(Ndomains,1);
pred_lengths = zeros(Ndomains,1);
no_obs_im = ones(size_1,size_2);
for dom = 1:Ndomains
    goal = all_states_xy{dom}(end,:);
    start = all_states_xy{dom}(1,:);
    optimal_lengths(dom) = length(all_states_xy{dom});
    im = reshape(all_im_data(dom,:),size_1,size_2);
    G = Gridworld_Graph8(im,goal(1),goal(2));
    G_no_obs = Gridworld_Graph8(no_obs_im,goal(1),goal(2));
    value_prior = reshape(all_value_data(dom,:),size_1,size_2);
    if ~isempty(all_states_xy{dom}) && size(all_states_xy{dom},1)>1
        L = size(all_states_xy{dom},1)*2;
        pred_traj = zeros(L,2);
        pred_traj(1,:) = all_states_xy{dom}(1,:);
        for j = 2:L
            % creat current state vector and image vector, and save to file
            state_xy_data = uint8([pred_traj(j-1,1)-1, pred_traj(j-1,2)-1]);
            im_data = uint8(reshape(im',1,[]));
            value_data = uint8(reshape(value_prior',1,[]));
            % call NN to predict action from input file 
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
                pred_lengths(dom) = j;
                break;
            end
        end
    end
    disp(Ndomains-dom);
end
end
