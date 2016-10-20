classdef Gridworld_Graph8 < Finite_MDP_class
    % Gridworld domain with obstacles. Actions are to
    % {n,s,e,w,ne,nw,se,sw}. Transitions are deterministic
    properties
        Nrow = 1;      % image rows
        Ncol = 1;      % image columns
        img = [];      % image
        obstacles = [];% indices of obtacles in image
        non_obstacles; % indices of obtacles in image
        targetRow = 1;
        targetCol = 1;
        G = [];        % transition graph
        W = [];
        state_map_col;   % map from states to col values
        state_map_row;   % map from states to row values
    end
    methods (Static)
        function [newrow,newcol] = north(row,col,Nrow,Ncol,im)
            newrow = max(row-1,1);
            newcol = col;
            if im(newrow,newcol) == 0   % obstacle
                newrow = row;
                newcol = col;
            end
        end
        function [newrow,newcol] = northeast(row,col,Nrow,Ncol,im)
            newrow = max(row-1,1);
            newcol = min(col+1,Ncol);
            if im(newrow,newcol) == 0   % obstacle
                newrow = row;
                newcol = col;
            end
        end
        function [newrow,newcol] = northwest(row,col,Nrow,Ncol,im)
            newrow = max(row-1,1);
            newcol = max(col-1,1);
            if im(newrow,newcol) == 0   % obstacle
                newrow = row;
                newcol = col;
            end
        end
        function [newrow,newcol] = south(row,col,Nrow,Ncol,im)
            newrow = min(row+1,Nrow);
            newcol = col;
            if im(newrow,newcol) == 0   % obstacle
                newrow = row;
                newcol = col;
            end
        end
        function [newrow,newcol] = southeast(row,col,Nrow,Ncol,im)
            newrow = min(row+1,Nrow);
            newcol = min(col+1,Ncol);
            if im(newrow,newcol) == 0   % obstacle
                newrow = row;
                newcol = col;
            end
        end
        function [newrow,newcol] = southwest(row,col,Nrow,Ncol,im)
            newrow = min(row+1,Nrow);
            newcol = max(col-1,1);
            if im(newrow,newcol) == 0   % obstacle
                newrow = row;
                newcol = col;
            end
        end        
        function [newrow,newcol] = east(row,col,Nrow,Ncol,im)
            newrow = row;
            newcol = min(col+1,Ncol);
            if im(newrow,newcol) == 0   % obstacle
                newrow = row;
                newcol = col;
            end
        end
        function [newrow,newcol] = west(row,col,Nrow,Ncol,im)
            newrow = row;
            newcol = max(col-1,1);
            if im(newrow,newcol) == 0   % obstacle
                newrow = row;
                newcol = col;
            end
        end
        function [rows,cols] = neighbors(row,col,Nrow,Ncol,im)
            [rows,cols] = Gridworld_Graph8.north(row,col,Nrow,Ncol,im);
            [newrow,newcol] = Gridworld_Graph8.south(row,col,Nrow,Ncol,im);
            rows = [rows,newrow]; cols = [cols,newcol];
            [newrow,newcol] = Gridworld_Graph8.east(row,col,Nrow,Ncol,im);
            rows = [rows,newrow]; cols = [cols,newcol];
            [newrow,newcol] = Gridworld_Graph8.west(row,col,Nrow,Ncol,im);
            rows = [rows,newrow]; cols = [cols,newcol];
            [newrow,newcol] = Gridworld_Graph8.northeast(row,col,Nrow,Ncol,im);
            rows = [rows,newrow]; cols = [cols,newcol];
            [newrow,newcol] = Gridworld_Graph8.northwest(row,col,Nrow,Ncol,im);
            rows = [rows,newrow]; cols = [cols,newcol];
            [newrow,newcol] = Gridworld_Graph8.southeast(row,col,Nrow,Ncol,im);
            rows = [rows,newrow]; cols = [cols,newcol];
            [newrow,newcol] = Gridworld_Graph8.southwest(row,col,Nrow,Ncol,im);
            rows = [rows,newrow]; cols = [cols,newcol];
        end
    end
    methods 
        function obj = Gridworld_Graph8(ImageFile,targetRow,targetCol)
            if ischar(ImageFile)
                % construct graph from image file
                im = imread(ImageFile);
                img = double(rgb2gray(im));
            else
                % image is already a matrix
                img = ImageFile;
            end
            Nrow = size(img,1);
            Ncol = size(img,2);
            obstacles = find(img == 0);
            non_obstacles = find(img ~= 0);
            target = sub2ind([Nrow,Ncol],targetRow,targetCol);
            Ns = Nrow*Ncol;
            Na = 8;
            Pn = zeros(Ns,Ns);  % north
            Ps = zeros(Ns,Ns);  % south
            Pe = zeros(Ns,Ns);  % east
            Pw = zeros(Ns,Ns);  % west
            Pne = zeros(Ns,Ns);  % north east
            Pnw = zeros(Ns,Ns);  % north west
            Pse = zeros(Ns,Ns);  % south east
            Psw = zeros(Ns,Ns);  % south west
            G = zeros(Ns,Ns);
            R = -1*ones(Ns,Na);
            R(:,5:8) = R(:,5:8)*sqrt(2); % diagonal cost
            R(target,:) = 0;
            for row = 1:Nrow
                for col = 1:Ncol
                    curpos = sub2ind([Nrow,Ncol],row,col);
                    [rows,cols] = Gridworld_Graph8.neighbors(row,col,Nrow,Ncol,img);
                    neighbor_inds = sub2ind([Nrow,Ncol],rows,cols);
                    Pn(curpos,neighbor_inds(1)) = Pn(curpos,neighbor_inds(1)) + 1;
                    Ps(curpos,neighbor_inds(2)) = Ps(curpos,neighbor_inds(2)) + 1;
                    Pe(curpos,neighbor_inds(3)) = Pe(curpos,neighbor_inds(3)) + 1;
                    Pw(curpos,neighbor_inds(4)) = Pw(curpos,neighbor_inds(4)) + 1;
                    Pne(curpos,neighbor_inds(5)) = Pne(curpos,neighbor_inds(5)) + 1;
                    Pnw(curpos,neighbor_inds(6)) = Pnw(curpos,neighbor_inds(6)) + 1;
                    Pse(curpos,neighbor_inds(7)) = Pse(curpos,neighbor_inds(7)) + 1;
                    Psw(curpos,neighbor_inds(8)) = Psw(curpos,neighbor_inds(8)) + 1;
                end
            end
            G = Pn | Ps | Pe | Pw | Pne | Pnw | Pse | Psw;
            W = max(max(max(max(max(max(max(Pn,Ps),Pe),Pw),sqrt(2)*Pne),sqrt(2)*Pnw),sqrt(2)*Pse),sqrt(2)*Psw);
            Pn = Pn(non_obstacles,:); Pn = Pn(:,non_obstacles);
            Ps = Ps(non_obstacles,:); Ps = Ps(:,non_obstacles);
            Pe = Pe(non_obstacles,:); Pe = Pe(:,non_obstacles);
            Pw = Pw(non_obstacles,:); Pw = Pw(:,non_obstacles);
            Pne = Pne(non_obstacles,:); Pne = Pne(:,non_obstacles);
            Pnw = Pnw(non_obstacles,:); Pnw = Pnw(:,non_obstacles);
            Pse = Pse(non_obstacles,:); Pse = Pse(:,non_obstacles);
            Psw = Psw(non_obstacles,:); Psw = Psw(:,non_obstacles);
            G = G(non_obstacles,:); G = G(:,non_obstacles);
            W = W(non_obstacles,:); W = W(:,non_obstacles);
            R = R(non_obstacles,:);
            P = cat(3,Pn,Ps,Pe,Pw,Pne,Pnw,Pse,Psw);
            obj@Finite_MDP_class(P,R);
            obj.Nrow = Nrow;
            obj.Ncol = Ncol;
            obj.img = img;
            obj.obstacles = obstacles;
            obj.non_obstacles = non_obstacles;
            obj.targetRow = targetRow;
            obj.targetCol = targetCol;
            obj.G = G;
            obj.W = W;
            [state_map_col, state_map_row] = meshgrid(1:Ncol,1:Nrow);
            obj.state_map_row = state_map_row(non_obstacles);
            obj.state_map_col = state_map_col(non_obstacles);
        end
        function [G,W] = getGraph(obj)
            % return directed graph G with weights W for gridworld
            G = sparse(double(obj.G));
            W = obj.W(obj.W~=0);
        end
        function [G,W] = getGraph_inv(obj)
            % return inverse directed graph G with weights W for gridworld
            G = sparse(double(obj.G'));
            W_inv = obj.W';
            W = W_inv(W_inv~=0);
        end
        function [im] = val2image(obj,val)
            % put values (for states) on the image
            im = zeros(obj.Nrow,obj.Ncol);
            im(obj.non_obstacles) = val;
        end
        function [im] = getValuePrior(obj)
            % get a prior for the value function (just Euclidean distance to goal)
            [s_map_col, s_map_row] = meshgrid(1:obj.Ncol,1:obj.Nrow);
            im = sqrt((s_map_col-obj.targetCol).^2 + (s_map_row-obj.targetRow).^2);
        end
        function [im] = getRewardPrior(obj)
            % get a prior for the reward function (just -1 for every non-goal state)
            im = -1*ones(obj.Nrow,obj.Ncol);
            im(obj.targetRow,obj.targetCol) = 10;
        end
        function [im] = getStateImage(obj, row, col)
            % get an image for the current state (just 0 for every other state)
            im = zeros(obj.Nrow,obj.Ncol);
            im(row,col) = 1;
        end
        function [s] = map_ind_to_state(obj,row,col)
            % find state index for given row and col
            s = find(obj.state_map_row == row & obj.state_map_col == col);
        end
        function [r,c] = getCoords(obj,states)
            [r,c] = ind2sub([obj.Nrow,obj.Ncol],obj.non_obstacles(states));
        end
        function [Nrow,Ncol] = getSize(obj)
            Nrow = obj.Nrow;
            Ncol = obj.Ncol;
        end
    end
end