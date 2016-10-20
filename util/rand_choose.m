function res = rand_choose(in_vec)
% sample an element from probability vector in_vec
if size(in_vec,2)==1
    in_vec = in_vec';
end

tmp = [0 cumsum(in_vec)];
q = rand;
res = find(q>tmp(1:end-1) & q<tmp(2:end),1);