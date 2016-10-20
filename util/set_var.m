function set_var(name,val)
% if variable doesn't already exist, set it to val
W = evalin('caller','who'); %or 'base'
doesexist=0;
for ii= 1:length(W)
    nm1=W{ii};
    doesexist=strcmp(nm1,name)+doesexist;
end
doesexist(doesexist>0)=1;
if ~doesexist
    assignin('caller',name,val);
end