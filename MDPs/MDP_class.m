classdef MDP_class < matlab.mixin.Copyable
    % Interface for MDP
    methods (Abstract)
        Ns= getNumStates(obj);  % total states
        a = getNumActions(obj); % total possible actions
        a = getActions(obj,s);  % actions at state s
        r = getReward(obj,s,a);
        p = nextStateProb(obj,s,a);
        snext = sampleNextState(obj,s,a);
    end
end
