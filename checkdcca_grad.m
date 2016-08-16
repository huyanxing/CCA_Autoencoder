  %data = rand(6,20);
  %[featurenum,samplenum] = size(data);
  %hiddenSize = 5;
  %visibleSize = featurenum;
  %theta = initializeParameters(hiddenSize, visibleSize);
  %sparsityParam = 0.05; 
  %desired average activation of the hidden units.
  %lambda = 3e-3;         
  %weight decay parameter       
  %beta = 5;              
  %weight of sparsity penalty term       
  %epsilon = 0.1;
  %K=1;
  %subFeatureNum = [2,2,2];
  
% [cost,grad] = SplitSparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data,subFeatureNum,K);
% Instructions:

%%
clc;
clear;
SettingParametersForChecking;
%%
%W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
%W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
%%
%[cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, sparsityParam, beta, data);

%numGrad = computeNumericalGradient( @(x)sparseAutoencoderCost(x, visibleSize, hiddenSize, lambda, sparsityParam,...
                                                                   %beta, data), theta);

%W2 = W2(:);
[cost,grad] = SplitSparseAutoencoderCost(theta,visibleSize, hiddenSize, lambda, sparsityParam, beta, data,subFeatureNum,gamma,K);

numGrad = computeNumericalGradient( @(x)SplitSparseAutoencoderCost(x,visibleSize, hiddenSize, lambda, sparsityParam,...
                                                         beta, data,subFeatureNum,gamma,K), theta);
%%

%%
  diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff);
    if diff < 1e-8,
        disp ('OK')
    else
        disp ('Difference too large. Check your gradient computation again')
    end