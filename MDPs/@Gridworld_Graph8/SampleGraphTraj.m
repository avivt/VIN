function [states_xy, states_one_hot] = SampleGraphTraj(M,Ns)
% sample Ns states trajectories from random nodes in graph object M to goal
% return states_xy: cell array of trajectories in xy-space
%        states_one_hot: cell array of trajectories in one-hot vectors for
%        x and y
[G,W] = M.getGraph_inv;
G_inv = G';     % transpose graph for tranposing single-node SP -> single destination SP
N = size(G,1);
if N >= Ns
    rand_ind = randperm(N);
else
    rand_ind = repmat(randperm(N),1,10);    % hack for small domains
end

init_states = rand_ind(1:Ns);
goal_s = M.map_ind_to_state(M.targetRow,M.targetCol);
states = cell(Ns,1);
states_xy = cell(Ns,1);
states_one_hot = cell(Ns,1);
i = 1;
options.edge_weight = W;
[~, pred] = shortest_paths(G_inv,goal_s,options);       % all SP from goal
for n = 1:Ns
    [path] = SP(pred,goal_s,init_states(n));    % get SP from goal->init
    path = path(end:-1:1)';                     % reverse path since we want init->goal
    states{i} = path;
    i = i+1;
end
for i = 1:length(states)
    L = length(states{i});
    [r,c] = M.getCoords(states{i});
    row_mat = zeros(L,M.Nrow);
    col_mat = zeros(L,M.Ncol);
    for j = 1:L
        row_mat(j,r(j)) = 1;
        col_mat(j,c(j)) = 1;
    end
    states_one_hot{i} = [row_mat col_mat];
    states_xy{i} = [r,c];
end