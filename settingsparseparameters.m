%%        
%           clc;
%           clear;
%           data1=rand(24,3);
%           data2=rand(24,3);
%           data3=rand(24,3);

          data1 = featureset_wind60;
          data2 = featureset_mslp;
          data3 = featureset_temperature;
          data = [data1,data2,data3];
          neuronnum = 12;
          subFeatureNum = [featurenum_wind,featurenum_mslp,featurenum_temperature];
%% deal with missing value
          

%%
          [samplenum,featurenum]=size(data);
          visibleSize = featurenum;
          hiddenSize  = neuronnum;
%floor(featurenum*0.5);
          sparsityParam = 0.05; % desired average activation of the hidden units.
          lambda = 0.000003;         % weight decay parameter       
          beta = 0;              % weight of sparsity penalty term       
          %epsilon = 0.1;	       % epsilon for ZCA whitening
%%
          theta=initializeParameters(hiddenSize, visibleSize);
          K= 10;
          [data,~] = scale(data);
          data = data';
         % SplitSparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                                          %  lambda, sparsityParam, beta, data,subFeatureNum,K)
          
          clearvars data1 data2 data3 