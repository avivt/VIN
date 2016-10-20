% script to generate testing data for learning trajectories
addpaths;
set_var('size_1',28); set_var('size_2',28);
dom_size = [size_1,size_2];                 % domain size
maxTrajLen = (size_1+size_2);             % this is approximate, just to preallocate memory
set_var('Ndomains', 1000);                 % number of domains
set_var('maxObs', 10);                      % maximum number of obstacles in a domain
set_var('maxObsSize',[]);                   % maximum obstacle size
Ntrajs=1;                       % trajectories from each domain
set_var('goal', [1,1]);                     % goal position
set_var('rand_goal', true);                % random goal position
set_var('add_border', true);               % add border (of abstacles) to domain
set_var('prior', 'value');                  % prior reward function
set_var('zero_min_action', true);          % actions from 0 to 7 instead of 1 to 8
set_var('state_batch_size', 1);            % batchsize for states per each data sample

numActions = 8;
action_vecs = ([[-1,0; 1,0; 0,1; 0,-1]; 1/sqrt(2)*[-1,1; -1,-1; 1,1; 1,-1]])';  % state difference unit vectors for each action
action_vecs_unnorm = ([-1,0; 1,0; 0,1; 0,-1; -1,1; -1,-1; 1,1; 1,-1]);          % un-normalized state difference vectors

% containers for data
all_im_data = zeros([Ndomains, size_1*size_2]);        % obstacle image
all_value_data = zeros([Ndomains, size_1*size_2]);     % value function prior
numDomains = 1;
all_states_xy = cell(Ndomains,1);
%% make data
for dom = 1:3*Ndomains                                        % loop over domains
    if numDomains > Ndomains
        break;
    end
    if rand_goal
        goal(1,1) = 1+randi(size_1-1);
        goal(1,2) = 1+randi(size_2-1);
    end
    obs = obstacle_gen(dom_size,goal,maxObsSize);           % generate random obstacles
    n_obs = obs.add_N_rand_obs(randi(maxObs));
    if add_border
        add_border_res = obs.add_border;
    else
        add_border_res = 0;
    end
    if n_obs == 0 || add_border_res
        continue;   % no obstacles added, or problem with border, skip
    end
    im = double(rgb2gray(obs.getimage));
    im = max(max(im)) - im; im = im./max(max(im));
    G = Gridworld_Graph8(im,goal(1),goal(2));               % make graph from obstacle map
    if strcmp(prior,'value')
        value_prior = G.getValuePrior;                           % get prior over value function (just distance to goal)
    elseif strcmp(prior,'reward')
        value_prior = G.getRewardPrior;                          % get prior over value function (just reward)
    end
    [states_xy, states_one_hot] = SampleGraphTraj(G,Ntrajs);% sample shortest-path trajectories in graph
    if isempty(states_xy{1}) || size(states_xy{1},1)==1
        continue;	% trivial trajectory, skip
    end
    % calculate the actions along the trajectory
    % an action corresponds to difference in state (n,s,e,w,ne,nw,se,sw)
    state_diff = diff(states_xy{1});                        % state difference
    norm_state_diff = state_diff.*repmat(1./sqrt(sum(state_diff.^2,2)),1,size(state_diff,2));        % normalized state difference
    prj_state_diff = norm_state_diff*action_vecs;           % project state difference on action vectors
    actions_one_hot = abs(prj_state_diff-1)<1e-5;           % action corresponds to projection==1
    actions = actions_one_hot * (1:numActions)';            % action labels
    ns = size(states_xy{1},1)-1;
    % add trajectory to dataset
    all_im_data(numDomains,:) = reshape(im,1,[]);
    all_value_data(numDomains,:) = reshape(value_prior,1,[]);
    all_states_xy{numDomains} = states_xy{1};
    numDomains = numDomains+1;
    disp(Ndomains - numDomains);
end
%% save data
disp('saving data');
set_var('data_dir', '~/Data/LearnTraj/');
set_var('data_file', 'data.mat');         % store training data variables
save([data_dir data_file],'all_im_data','all_value_data',...
    'all_states_xy');
% save([data_dir env_file],'size_1','size_2','dom_size','goal','numActions','action_vecs_unnorm','all_states_xy');
