classdef obstacle_gen < handle
    % class for generating obstacles in a domain
    properties
        domsize     % domain size (pixels in each dimension)
        mask        % forbidden area for obstacles (e.g., goal position) specified as list of image coordinates
        dom         % domain image
        obs_types   % list of obstacle types
        num_types   % number of available obstacle types
        size_max    % maximum obstacle size
    end
    methods
        function obj = obstacle_gen(domsize,mask,size_max)
            obj.domsize = domsize;
            obj.dom = zeros(domsize);
            obj.mask = mask;
            obj.obs_types = {'circ';'rect'};
            obj.num_types = length(obj.obs_types);
            if nargin<3
                obj.size_max = max(obj.domsize)/4;
            else
                if ~isempty(size_max)
                    obj.size_max = size_max;
                else
                    obj.size_max = max(obj.domsize)/4;
                end
            end
        end
        function [cond] = check_mask(obj,dom)
            % check if an obstacle is in the mask region
            ind = sub2ind(size(obj.dom),obj.mask(:,1),obj.mask(:,2));
            if nargin < 2
                cond = any(obj.dom(ind));
            else
                cond = any(dom(ind));
            end
        end
        function [res] = add_rand_obs(obj,type)
            % add an obstacle of type ('circ','rect') at a random position,
            % and random size. Returns 0 on success, and 1 otherwise.
            % maximal object size is 1/4 of domain size
            
            if strcmp(type,'circ')
                rand_rad = ceil(rand*obj.size_max);
                randx = ceil(rand*obj.domsize(1));
                randy = ceil(rand*obj.domsize(1));
                im_try = insertShape(obj.dom, 'FilledCircle', [randx, randy, rand_rad], 'LineWidth', 1,'Opacity',1,'SmoothEdges',false);
            elseif strcmp(type,'rect')
                rand_hgt = ceil(rand*obj.size_max);
                rand_wid = ceil(rand*obj.size_max);
                randx = ceil(rand*obj.domsize(1));
                randy = ceil(rand*obj.domsize(1));
                im_try = insertShape(obj.dom, 'FilledRectangle', [randx, randy, rand_wid, rand_hgt], ...
                                     'LineWidth', 1,'Opacity',1,'SmoothEdges',false);
            end
            if obj.check_mask(im_try)
                res = 1;
                return;
            else
                obj.dom = im_try;
                res = 0;
                return;
            end
        end
        function [res] = add_N_rand_obs(obj,N)
            % add N random obstacles of random types. Returns number of
            % obstacles actually added (due to masking).
            res = 0;
            for i = 1:N
                rand_type = obj.obs_types{randi(obj.num_types)};
                [t] = obj.add_rand_obs(rand_type);
                if t==0, res = res+1; end
            end
        end
        function [im] = getimage(obj)
            im = obj.dom;
        end
        function [im] = getimage3D(obj)
            f = figure;
            for i = 1:obj.domsize(1)
                for j = 1:obj.domsize(2)
                    if obj.dom(i,j) > 0
                        p = cube(obj, i, j, 0, 1);
%                         p.FaceColor = 'interp';
%                         p.FaceLighting = 'gouraud';
                    end
                end
            end
            view(3);
        end
        function [res] = add_border(obj)
            im_try = insertShape(obj.dom, 'Rectangle', [1, 1, obj.domsize(1), obj.domsize(2)], ...
                                     'LineWidth', 1,'Opacity',1,'SmoothEdges',false);
            if obj.check_mask(im_try)
                res = 1;
            else
                obj.dom = im_try;
                res = 0;
            end
        end
        function p = cube(obj, X0, Y0, Z0, C0)
            X1 = [0;0;1;1], Y1 = [0;1;1;0], Z1 = [0;0;0;0];
            X2 = [0;0;1;1], Y2 = [0;1;1;0], Z2 = [1;1;1;1];
            Y3 = [0;0;1;1], Z3 = [0;1;1;0], X3 = [0;0;0;0];
            Y4 = [0;0;1;1], Z4 = [0;1;1;0], X4 = [1;1;1;1];
            X5 = [0;0;1;1], Z5 = [0;1;1;0], Y5 = [0;0;0;0];
            X6 = [0;0;1;1], Z6 = [0;1;1;0], Y6 = [1;1;1;1];
            X = [X1,X2,X3,X4,X5,X6] + X0;
            Y = [Y1,Y2,Y3,Y4,Y5,Y6] + Y0;
            Z = [Z1,Z2,Z3,Z4,Z5,Z6] + Z0;
            C = C0*rand(size(X));
            p = patch(X,Y,Z,C);
        end
    end
end
