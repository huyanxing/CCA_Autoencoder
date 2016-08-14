%% prepare data
load('weatherdata.mat')
featurenum_mslp =12; % the features
featurenum_temperature =12;
featurenum_wind =12;
data_mslp = mslp(:,3);
data_temperature = temp(:,3);
data_wind10_ini = wind10(:,[3,4]);
data_wind60_ini = wind60(:,[3,4]);
clearvars mslp temp wind10 wind60 dewpoints;
move = 3;%time slots

[featureset_mslp, labelset_mslp] = TimeSerisFormat(data_mslp, featurenum_mslp,move);
[featureset_temperature, labelset_temperature] = TimeSerisFormat(data_temperature, featurenum_temperature,move);

% onlyspeed = 0;
% if onlyspeed == 1
%     data_wind10 = data_wind10_ini(:,2);
% else
%     data_wind10 = data_wind10_ini(:,2).* sind(data_wind10_ini(:,1));
%     for i=1:max(size(data_wind10))
%         if isnan(data_wind10(i)) == 1
%         data_wind10(i) = 0.5*(data_wind10(i-1) + data_wind10(i+1));
%         %data_wind10(i) = 0;
%         end
%    end
% end     
% 
% [featureset_wind10 , labelset_wind10 ] = TimeSerisFormat(data_wind10 , featurenum_wind,move);



for i=1:max(size(data_wind60_ini))
    if isnan(data_wind60_ini(i,1)) == 1
        data_wind60_ini(i,1) = 0.5*(data_wind60_ini(i-1,1) + data_wind60_ini(i+1,1));
    end
    if isnan(data_wind60_ini(i,1)) == 1
        data_wind60_ini(i,2) = 0.5*(data_wind60_ini(i-1,2) + data_wind60_ini(i+1,2));
    end
end
data_wind60 = data_wind60_ini(:,2).* sind(data_wind60_ini(:,1));

[featureset_wind60 , labelset_wind60 ] = TimeSerisFormat(data_wind60 , featurenum_wind,move);
%% deal with missing values
label = labelset_wind60;
Totalset = [label,featureset_wind60,featureset_mslp,featureset_temperature];
Totalset(any(isnan([label,featureset_wind60]),2),:)= [];
label = Totalset(:,1);
featureset_wind60 = Totalset(:,2:featurenum_wind+1);
featureset_mslp = Totalset(:,featurenum_wind+2:featurenum_mslp+featurenum_wind+1);
featureset_temperature = Totalset(:,featurenum_mslp+featurenum_wind+2:end);

clearvars data_wind60_ini Totalset;
%%