%%        
%           clc;
%           clear;
          data1=rand(24,3);
          data2=rand(24,3);
          data3=rand(24,3);


          data = [data1,data2,data3];
          neuronnum = 3;
          subFeatureNum = [3,3,3];
 
%%
          [samplenum,featurenum]=size(data);
          visibleSize = featurenum;
          hiddenSize  = neuronnum;
%floor(featurenum*0.5);
          gamma = -0.003;
          sparsityParam = 0.05; % desired average activation of the hidden units.
          lambda = 0.000003;         % weight decay parameter       
          beta = 0.001;              % weight of sparsity penalty term       
          %epsilon = 0.1;	       % epsilon for ZCA whitening
%%
          theta=initializeParameters(hiddenSize, visibleSize);
          K= 2;
          [data,~] = scale(data);
          data = data';
         % SplitSparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                                          %  lambda, sparsityParam, beta, data,subFeatureNum,K)
          
          clearvars data1 data2 data3 