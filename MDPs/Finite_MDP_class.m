classdef Finite_MDP_class < MDP_class
    % Finite state and action MDP
    properties
        P = [];     % transition kernel
        R = [];     % reward
        A = [];     % possible actions at each state
        Ns = 0;     % number of states
        Na = 0;     % number of actions
    end
    methods 
        function obj = Finite_MDP_class(P,R,A)
            % constructor:
            % P is Ns*Ns*Na matrix of transitions P(s'|s,a)
            % R is Ns*Na matrix of deterministic rewards r(s,a)
            % A is Ns*Na binary matrix of available actions at each state
            %   (default - all actions are possible).
            obj.P = P;
            obj.R = R;
            obj.Ns = size(P,1);
            obj.Na = size(P,3);
            if nargin < 3
                A = ones(obj.Ns,obj.Na);
            end
            obj.A = A;
        end
        
        function Ns= getNumStates(obj)
            Ns = obj.Ns;
        end
        
        function Na = getNumActions(obj)
            Na = obj.Na;
        end
        
        function a = getActions(obj,s)
            a = find(obj.A(s,:));
        end
        
        function r = getReward(obj,s,a)
            r = obj.R(s,a)';
        end
        
        function p = nextStateProb(obj,s,a)
            % get next state probability for action a
            % if a is a scalar the function returns a row vector
            % if a is a vector then a matrix is returned with the
            % probabilities on rows
            if numel(a) == 1
                p = squeeze(obj.P(s,:,a));
            else
                p = squeeze(obj.P(s,:,a))';
            end
        end
        
        function snext = sampleNextState(obj,s,a)
            % sample a next state given s and a
            snext = rand_choose(obj.nextStateProb(s,a));
        end
    end
end
