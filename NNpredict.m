function [y] = NNpredict(nn,im,value,x,y,maxX,maxY)
% call python to generate prediction for nn object, with input image and
% x,y state (0<x<=maxX,0<y<=maxY)

% creat one-hot current state vector and image vector, and save to
% file
r_one_hot = zeros(1,maxX);
c_one_hot = zeros(1,maxY);
r_one_hot(x) = 1;
c_one_hot(y) = 1;
state_data = uint8([r_one_hot, c_one_hot]);
state_xy_data = uint8([x-1, y-1]);
im_data = uint8(reshape(im',1,[]));
value_data = uint8(reshape(value',1,[]));
% call NN to predict action from input file (don't know how to
% pass state data directly yet)
save('test_input.mat','im_data','state_data','value_data','state_xy_data');
y = nn.predict(pyargs('input', 'test_input.mat'));
y = max(y,1);
