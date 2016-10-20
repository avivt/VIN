% script to generate training data for learning trajectories
% The data is organized in batches of multiple states from the same domain.
% The batch size is determined by state_batch_size.
% In addition, a flattened data (non-batched) organization is maintained.

addpaths;
% set parameters (defaults)
set_var('size_1',28); set_var('size_2',28);
dom_size = [size_1,size_2];                 % domain size
maxTrajLen = (size_1+size_2);               % this is approximate, just to preallocate memory
set_var('Ndomains', 10000);                 % number of domains
set_var('maxObs', 10);                      % maximum number of obstacles in a domain
set_var('maxObsSize',[]);                   % maximum obstacle size
set_var('Ntrajs', 1);                       % trajectories from each domain
set_var('goal', [1,1]);                     % goal position
set_var('rand_goal', false);                % random goal position
set_var('state_batch_size', 1);             % batchsize for states per each data sample

% containers for flattened data
maxSamples = Ndomains*Ntrajs*maxTrajLen/2;                                  % this is approximate, just to preallocate memory
im_data = uint8(zeros([maxSamples, size_1*size_2]));                        % obstacle image
value_data = uint8(zeros([maxSamples, size_1*size_2]));                     % value function prior (e.g., a reward function)
state_onehot_data = uint8(zeros([maxSamples, size_1+size_2]));              % 1-hot vectors of position for each dimension (x,y)
state_xy_data = uint8(zeros([maxSamples, 2]));                              % position (in both coordinates)
label_data = uint8(zeros([maxSamples, 1]));                                 % action

% containers for batched data
numSamples = 1;
all_states_xy = cell(Ndomains*Ntrajs,1);
all_doms = cell(Ndomains*Ntrajs,1);
numTrajs = 1;
maxBatches = ceil(Ndomains*Ntrajs*maxTrajLen/state_batch_size);
numBatches = 1;
batch_im_data = uint8(zeros([maxBatches, size_1*size_2]));        % obstacle image
batch_value_data = uint8(zeros([maxBatches, size_1*size_2]));     % value function prior
state_x_data = uint8(zeros([maxBatches, state_batch_size]));      % position (in 1st coordinate)
state_y_data = uint8(zeros([maxBatches, state_batch_size]));      % position (in 2nd coordinate)
batch_label_data = uint8(zeros([maxBatches, state_batch_size]));  % action

%% make data
figure;
dom = 1;
while dom <= Ndomains
    % allocate buffers for batched data from this domain 
    s1_buffer = uint8(zeros([ceil(Ntrajs*maxTrajLen/state_batch_size), 1]));
    s2_buffer = uint8(zeros([ceil(Ntrajs*maxTrajLen/state_batch_size), 1]));
    label_buffer = uint8(zeros([ceil(Ntrajs*maxTrajLen/state_batch_size), 1]));
    % generate random domain
    buffer_pos = 1;
    if rand_goal
        goal(1,1) = 1+randi(size_1-1);
        goal(1,2) = 1+randi(size_2-1);
    end
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
    hold on;
    for i = 1:Ntrajs     % loop over trajectories in domain
        if ~isempty(states_xy{i}) && size(states_xy{i},1)>1
            % calculate the actions along the trajectory
            actions = extract_action(states_xy{i});
            ns = size(states_xy{i},1)-1;
            % add trajecory to dataset
            % we transpose - since python is row major order
            % we subtract 1 - since python indexing starts at zero
            im_data(numSamples:numSamples+ns-1,:) = repmat(reshape(im',1,[]),ns,1);  
            value_data(numSamples:numSamples+ns-1,:) = repmat(reshape(value_prior',1,[]),ns,1);
            state_onehot_data(numSamples:numSamples+ns-1,:) = states_one_hot{i}(1:ns,:);
            state_xy_data(numSamples:numSamples+ns-1,:) = states_xy{i}(1:ns,:)-1;
            s1_buffer(buffer_pos:buffer_pos+ns-1,:) = states_xy{i}(1:ns,1)-1;
            s2_buffer(buffer_pos:buffer_pos+ns-1,:) = states_xy{i}(1:ns,2)-1;
            label_data(numSamples:numSamples+ns-1,:) = actions - 1;
            label_buffer(buffer_pos:buffer_pos+ns-1,:) = actions - 1;
            % update sample counters and flattened data containers
            numSamples = numSamples+ns;
            buffer_pos = buffer_pos+ns;
            all_states_xy{numTrajs} = states_xy{i};
            all_doms{numTrajs} = uint8(im);
            numTrajs = numTrajs + 1;
            % plot
            plot(states_xy{i}(:,2),states_xy{i}(:,1));drawnow;
        end
    end
    % batch size is fixed. We replicate the last sample to fill the batch.
    if mod(buffer_pos-1,state_batch_size)~=0
        samples_to_fill = state_batch_size-mod(buffer_pos,state_batch_size);
        s1_buffer(buffer_pos : buffer_pos+samples_to_fill) = s1_buffer(buffer_pos-1);
        s2_buffer(buffer_pos : buffer_pos+samples_to_fill) = s2_buffer(buffer_pos-1);
        label_buffer(buffer_pos : buffer_pos+samples_to_fill) = label_buffer(buffer_pos-1);
        buffer_pos = buffer_pos+samples_to_fill+1;
    end
    % fill data containers with random permutation of the data
    s1_buffer = s1_buffer(1:buffer_pos-1);
    s2_buffer = s2_buffer(1:buffer_pos-1);
    label_buffer = label_buffer(1:buffer_pos-1);
	rand_ind = randperm(buffer_pos-1);
	s1_buffer = s1_buffer(rand_ind);
	s2_buffer = s2_buffer(rand_ind);
	label_buffer = label_buffer(rand_ind);
    s1_batch = reshape(s1_buffer,state_batch_size,[])';
    s2_batch = reshape(s2_buffer,state_batch_size,[])';
    label_batch = reshape(label_buffer,state_batch_size,[])';
    cur_batch_size = size(s1_batch,1);
    state_x_data(numBatches:numBatches+cur_batch_size-1,:) = s1_batch;
    state_y_data(numBatches:numBatches+cur_batch_size-1,:) = s2_batch;
    batch_label_data(numBatches:numBatches+cur_batch_size-1,:) = label_batch;
    batch_im_data(numBatches:numBatches+cur_batch_size-1,:) = repmat(reshape(im',1,[]),cur_batch_size,1); 
    batch_value_data(numBatches:numBatches+cur_batch_size-1,:) = repmat(reshape(value_prior',1,[]),cur_batch_size,1); 
    numBatches = numBatches+cur_batch_size;
	%     pause;
    disp([num2str(Ndomains - dom) ' remaining domains']);
    hold off;
    dom = dom + 1;
end
% remove empty (preallocated) space in containers
im_data = im_data(1:numSamples-1,:);
value_data = value_data(1:numSamples-1,:);
state_onehot_data = state_onehot_data(1:numSamples-1,:);
state_xy_data = state_xy_data(1:numSamples-1,:);
label_data = label_data(1:numSamples-1,:);
all_states_xy = all_states_xy(1:numTrajs-1);
all_doms = all_doms(1:numTrajs-1);
state_x_data = state_x_data(1:numBatches-1,:);
state_y_data = state_y_data(1:numBatches-1,:);
batch_label_data = batch_label_data(1:numBatches-1,:);
batch_im_data = batch_im_data(1:numBatches-1,:);
batch_value_data = batch_value_data(1:numBatches-1,:);
%% save data
disp('saving data');
set_var('data_dir', '~/Data/LearnTraj/');
set_var('data_file', 'data.mat');          % store training data variables
save([data_dir data_file],'im_data','state_onehot_data','label_data','value_data',...
    'state_xy_data','state_x_data','state_y_data','batch_label_data','batch_im_data','batch_value_data');