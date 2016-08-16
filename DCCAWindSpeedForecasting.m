  %% prepare datasets;
    DataPrepareCCA;% path is :./pareparedata.m
    %% initialize paremeters
    SettingParametersSplitAE;% path is:./settingsparseparameters.m
    %% start training
    netparameters = SplitAEtrain(data,visibleSize,hiddenSize,sparsityParam,lambda ,beta,subFeatureNum,gamma,K);
    opt_feature = getOptFeatures(netparameters.W,netparameters.b, data);

    Num_cross = 10;
    opt_feature = opt_feature';
    Dataset_for_CrossPart =[label,opt_feature,data',featureset_wind60];
    [ Trainset, Testset ] = NCrossPart (Dataset_for_CrossPart,Num_cross);
    %% using svm for the forecasting
    cross = 1;

    for i = 1:cross
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

        fprintf('\n')   
        fprintf ('start train with fushion features...\n') 
        model_opt = svmtrain(train_label, train_opt_feature, '-s 4');
        fprintf('\n')   
        fprintf ('start prediction...\n') 
        [estlabel_opt{i}] = svmpredict(test_label,test_opt_feature,model_opt);

        fprintf('\n')   
        fprintf ('start train with no-fushion all features...\n') 
        model_all_together_feature = svmtrain(train_label, train_all_together_feature, '-s 4');
        fprintf('\n')   
        fprintf ('start prediction...\n') 
        [estlabel_all_together{i}] = svmpredict(test_label,test_all_together_feature,model_all_together_feature);

        fprintf('\n')   
        fprintf ('start train with wind_only features...\n') 
        model_only_wind = svmtrain(train_label, train_only_wind_feature, '-s 4');
        fprintf('\n')   
        fprintf ('start prediction...\n') 
        [estlabel_only_wind{i}] = svmpredict(test_label,test_only_wind_feature,model_only_wind);

        fprintf('\n')   
        fprintf ('finish a cross\n') 
        fprintf('\n')   
        % estlabel=estlabel';
        %Rsquare=accurancy(3);
        %Rsquare_vec(i)=Rsquare;
        %results = [results;estlabel];
    end