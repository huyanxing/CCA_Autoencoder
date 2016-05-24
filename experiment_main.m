clc;
clear;

addpath('./libsvm-3.21/matlab/')
addpath('./minFunc/' )
%% prepare datasets;
preparedata;% path is :./pareparedata.m
%% initialize paremeters
settingsparseparameters;% path is:./settingsparseparameters.m
%% start training
netparameters = SplitAEtrain(data,visibleSize,hiddenSize,sparsityParam,lambda ,beta,subFeatureNum,K);
opt_feature = getOptFeatures(netparameters.W,netparameters.b, data);

Num_cross = 2;
opt_feature = opt_feature';
Dataset_for_CrossPart =[label,opt_feature,data',featureset_wind60];
[ Trainset, Testset ] = NCrossPart (Dataset_for_CrossPart,Num_cross);
%% using svm for the forecasting


for i = 1:Num_cross
    train_total = Trainset{1,i};
    test_total = Testset{1,i};
    
    train_label = train_total(:,1);
    test_label = test_total(:,1);
    
    train_opt_feature = train_total(:,2:1+hiddenSize);
    train_all_together_feature = train_total(:,2+hiddenSize:1+hiddenSize+visibleSize);
    train_only_wind_feature = train_total(:,2+hiddenSize+visibleSize:1+hiddenSize+visibleSize+featurenum_wind);

    test_opt_feature = test_total(:,2:1+hiddenSize);
    test_all_together_feature = test_total(:,2+hiddenSize:1+hiddenSize+visibleSize);
    test_only_wind_feature = test_total(:,2+hiddenSize+visibleSize:1+hiddenSize+visibleSize+featurenum_wind);

    fprintf ('start train with fushion features...\n') 
    model_opt = svmtrain(train_label, train_opt_feature, '-s 3');
    fprintf ('start prediction...\n') 
    [estlabel_opt{i},accurancy_opt{i}] = svmpredict(test_label,test_opt_feature,model_opt);
    
    fprintf ('start train with fushion features...\n') 
    model_all_together_feature = svmtrain(train_label, train_all_together_feature, '-s 3');
    fprintf ('start prediction...\n') 
    [estlabel_all_together{i},accurancy_all_together{i}] = svmpredict(test_label,test_all_together_feature,model_all_together_feature);
   
    fprintf ('start train with fushion features...\n') 
    model_only_wind = svmtrain(train_label, train_only_wind_feature, '-s 3');
    fprintf ('start prediction...\n') 
    [estlabel_only_wind{i},accurancy_only_wind{i}] = svmpredict(test_label,test_only_wind_feature,model_only_wind);
    
    % estlabel=estlabel';
    %Rsquare=accurancy(3);
    %Rsquare_vec(i)=Rsquare;
    %results = [results;estlabel];
end

save('result.mat');