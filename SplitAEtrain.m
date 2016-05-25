function [netparameters] = SplitAEtrain( data,visibleSize,hiddenSize,sparsityParam,lambda ,beta,subFeatureNum,K )
%% debug process
debug = false;
if debug == true
    debugHiddenSize = 5;
    debugvisibleSize = 8;
    patches = rand([8 10]);
    subPatchnum = [3,2,3];
    theta = initializeParameters(debugHiddenSize, debugvisibleSize); 
    [cost, grad] = SplitSparseAutoencoderCost(theta, debugvisibleSize, debugHiddenSize, ...
                                           lambda, sparsityParam, beta, ...
                                           patches,subPatchnum,K);
                               
    numGrad = computeNumericalGradient( @(x) SplitSparseAutoencoderCost(x, debugvisibleSize, debugHiddenSize, ...
                                                  lambda, sparsityParam, beta, ...
                                                  patches,subPatchnum,K), theta);
   disp([numGrad grad]); 
   diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff);
   assert(diff < 1e-9, 'Difference too large. Check your gradient computation again');
end
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
options.maxIter = 5000;
options.display = 'on';
[optTheta, cost] = minFunc( @(p) SplitSparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, data,subFeatureNum,K), ...
                              theta, options);

%fprintf('Saving learned features and preprocessing matrices...\n');                          
%save('STL10Features.mat', 'optTheta', 'ZCAWhite', 'meanPatch');
%fprintf('Saved\n');

%% get the output features

netparameters.W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
netparameters.b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

%optfeatures= netparameter.W*data;

end


