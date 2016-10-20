function m = python_ndarray_to_matrix(p,psize)
cP = cell(p);
flat = cP{1,1}.flatten();
flatlist = flat.tolist();
m = zeros(psize);
ind = 1;
for i = 1:psize(1)
    for j = 1:psize(2)
        m(i,j) = double(flatlist{ind});
        ind = ind+1;
    end
end