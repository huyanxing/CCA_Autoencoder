          data = featureset;
          neuronnum = 12;
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
          %theta=initializeParameters(hiddenSize, visibleSize);
          [data,~] = scale(data);
          data = data';