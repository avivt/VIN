function [states_xy, states_one_hot] = OptimalActionsOnPath(M,traj)
% returns the optimal next states (shortest distance to goal) along path in
% xy-space
% return states_xy: cell array of trajectories in xy-space
%        states_one_hot: cell array of trajectories in one-hot vectors for
%        x and y
[G,W] = M.getGraph_inv;
G_inv = G';     % transpose graph for tranposing single-node SP -> single destination SP
% [dist] = all_shortest_paths(G);
N = size(G,1);
Ns = size(traj,1);
goal_s = M.map_ind_to_state(M.targetRow,M.targetCol);
states = zeros(Ns,1);
states_xy = zeros(Ns,2);
r_one_hot = zeros(Ns,M.Nrow);
c_one_hot = zeros(Ns,M.Ncol);
options.edge_weight = W;
[~, pred] = shortest_paths(G_inv,goal_s,options);       % all SP from goal
for s = 1:Ns
    curr_s = M.map_ind_to_state(traj(s,2),traj(s,1)); % TODO - figure out why?
    next_s = pred(curr_s);
    if next_s == 0
        next_s = curr_s;
    end
    [r,c] = M.getCoords(next_s);
    states(s) = next_s;
    states_xy(s,:) = [r,c];
    r_one_hot(s,r) = 1;
    c_one_hot(s,c) = 1;
end
states_one_hot = [r_one_hot, c_one_hot];