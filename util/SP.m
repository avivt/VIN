function [path] = SP(pred,s,t)
% trace-back shortest path from s(ource) to t(arget), with predecessor list
% pred, calculated before-hand
max_len = 1e3;
path = zeros(max_len,1);
i = max_len;
path(i) = t;
while path(i)~=s && i>1
    try
        path(i-1) = pred(path(i));
        i = i-1;
    catch
        warning('no path found, continuing');
        path = [];
        return
    end
end
if i>=1
    path = path(i:end);
else
    path = NaN;
end
end