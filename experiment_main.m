clc;
clear;

addpath('./libsvm-3.21/matlab/')
addpath('./minFunc/' )
addpath('./CRBM')

isCCA=0;

%% DCCA for wind speed

if isCCA == 1;
    DCCAWindSpeedForecasting;
end

%% No DCCA
isAE = 1;
isCRBM =1;
Num_cross = 10;
cross = 1; % the number of runed cross 
%% CRBM
datatype = 'temp';
featurenum_mslp = 12;
featurenum_temp = 12;
featurenum_wind = 12;
move = 3;
DataprepareAEandCRBM;

if isAE == 1
    SettingParametersAE;
    netparameters = AEtrain(data,visibleSize,hiddenSize,sparsityParam,lambda ,beta);
    opt_feature = getOptFeatures(netparameters.W,netparameters.b, data);
    Num_cross = 10;
    opt_feature = opt_feature';
    Dataset_for_CrossPart =[label,opt_feature,data'];
    [ Trainset, Testset ] = NCrossPart (Dataset_for_CrossPart,Num_cross);
    %% using svm for the forecasting
    cross = 1;
    for i = 1:cross
        train_total = Trainset{1,i};
        test_total = Testset{1,i};

        train_label = train_total(:,1);
        test_label = test_total(:,1);

        train_opt_feature = train_total(:,2:1+hiddenSize);
        train_ini_feature = train_total(:,2+hiddenSize:end);

        test_opt_feature = test_total(:,2:1+hiddenSize);
        test_ini_feature = test_total(:,2+hiddenSize:end);


        fprintf('\n')  
        fprintf('Data set : %s \n',datatype);
        fprintf('\n')  
        fprintf ('start train with AE optimized features...\n') 
        model_opt = svmtrain(train_label, train_opt_feature, '-s 4');
        fprintf('\n')   
        fprintf ('start prediction...\n') 
        [estLabel_opt{i}] = svmpredict(test_label,test_opt_feature,model_opt);

        fprintf('\n')   
        fprintf ('start train with  initial features...\n') 
        model_ini_feature = svmtrain(train_label, train_ini_feature, '-s 4');
        fprintf('\n')   
        fprintf ('start prediction...\n') 
        [estLabel_ini{i}] = svmpredict(test_label,test_ini_feature,model_ini_feature);

        fprintf('\n')   
        fprintf ('finish a cross\n') 
        fprintf('\n')   
        % estlabel=estlabel';
        %Rsquare=accurancy(3);
        %Rsquare_vec(i)=Rsquare;
        resultsLable{i}.label = test_label;
        resultsLable{i}.estLabel_opt = estLabel_opt{i};
        resultsLable{i}.estLabel_ini = estLabel_ini{i};
    end
end

if isCRBM == 1    
    hiddenSize= 5;    
    [model_opt,error]= crbmBB(featureset,hiddenSize);
    opt_feature = model_opt.top;
    Dataset_for_CrossPart =[label,opt_feature,featureset];
    [ Trainset, Testset ] = NCrossPart (Dataset_for_CrossPart,Num_cross);
    for i = 1:cross
          train_total = Trainset{1,i};
          test_total = Testset{1,i};

          train_label = train_total(:,1);
          test_label = test_total(:,1);

          train_opt_feature = train_total(:,2:1+hiddenSize);
          train_ini_feature = train_total(:,2+hiddenSize:end);

          test_opt_feature = test_total(:,2:1+hiddenSize);
          test_ini_feature = test_total(:,2+hiddenSize:end);


          fprintf('\n')  
          fprintf('Data set : %s \n',datatype);
          fprintf('\n')  
          fprintf ('start train with CRBM optimized features...\n') 
          model_opt = svmtrain(train_label, train_opt_feature, '-s 4');
          fprintf('\n')   
          fprintf ('start prediction...\n') 
          [estLabel_opt{i}] = svmpredict(test_label,test_opt_feature,model_opt);

          fprintf('\n')   
          fprintf ('start train with initial features...\n') 
          model_ini_feature = svmtrain(train_label, train_ini_feature, '-s 4');
          fprintf('\n')   
          fprintf ('start prediction...\n') 
          [estLabel_ini{i}] = svmpredict(test_label,test_ini_feature,model_ini_feature);

          fprintf('\n')   
          fprintf ('finish a cross\n') 
          fprintf('\n')   
            % estlabel=estlabel';
            %Rsquare=accurancy(3);
            %Rsquare_vec(i)=Rsquare;
          resultsLable{i}.label = test_label;
          resultsLable{i}.estLabel_opt = estLabel_opt{i};
          resultsLable{i}.estLabel_ini = estLabel_ini{i};
     end
end
save('result.mat');