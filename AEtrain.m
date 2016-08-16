function [netparameters] = AEtrain( data,visibleSize,hiddenSize,sparsityParam,lambda ,beta)
%% debug process
sample_num=max(size(data));
%% Apply preprocessing

% zcamark = 0;
% if zcamark == 1
% meandata = mean(data, 2);  
% patches = bsxfun(@minus, data, meandata);
% 
% % Apply ZCA whitening
% sigma = data * data' / numPatches;
% [u, s, v] = svd(sigma);
% ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
% data = ZCAWhite * data;
% end 
%%  Learn features
theta = initializeParameters(hiddenSize, visibleSize);
% Use minFunc to minimize the function
addpath minFunc/
options = struct;
options.Method = 'lbfgs'; 
options.maxIter = 1;
options.display = 'on';
[optTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, data), ...
                              theta, options);

%fprintf('Saving learned features and preprocessing matrices...\n');                          
%save('STL10Features.mat', 'optTheta', 'ZCAWhite', 'meanPatch');
%fprintf('Saved\n');

%% get the output features

netparameters.W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
netparameters.b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

%optfeatures= netparameter.W*data;

end


