function [actions] = extract_action(traj)
% extract actions from state trajectory
% an action corresponds to difference in state (n,s,e,w,ne,nw,se,sw)
numActions = 8;
action_vecs = ([[-1,0; 1,0; 0,1; 0,-1]; 1/sqrt(2)*[-1,1; -1,-1; 1,1; 1,-1]])';  % state difference unit vectors for each action
% action_vecs_unnorm = ([-1,0; 1,0; 0,1; 0,-1; -1,1; -1,-1; 1,1; 1,-1]);          % un-normalized state difference vectors

state_diff = diff(traj);                                                                    % state difference 
norm_state_diff = state_diff.*repmat(1./sqrt(sum(state_diff.^2,2)),1,size(state_diff,2));   % normalized state difference
prj_state_diff = norm_state_diff*action_vecs;           % project state difference on action vectors
actions_one_hot = abs(prj_state_diff-1)<1e-5;           % action corresponds to projection==1
actions = actions_one_hot * (1:numActions)';            % action labels