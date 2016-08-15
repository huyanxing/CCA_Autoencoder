    load('weatherdata.mat')
    switch datatype
        case 'mslp'
            feature_num = featurenum_mslp;
            data =  mslp(:,3);
        case 'temp'
            feature_num = featurenum_temp;
            data = temp(:,3);
        case 'wind10'
            feature_num = featurenum_wind;
            data_wind10_ini = wind10(:,[3,4]);
            for i=1:max(size(data_wind10_ini))
                if isnan(data_wind10_ini(i,1)) == 1
                    data_wind10_ini(i,1) = 0.5*(data_wind10_ini(i-1,1) + data_wind10_ini(i+1,1));
                end
                if isnan(data_wind10_ini(i,1)) == 1
                    data_wind10_ini(i,2) = 0.5*(data_wind10_ini(i-1,2) + data_wind10_ini(i+1,2));
                end
            end
            data = data_wind10_ini(:,2).* cosd(data_wind10_ini(:,1));
        case 'wind60'
            feature_num = featurenum_wind;
            data_wind60_ini = wind60(:,[3,4]);   
            for i=1:max(size(data_wind60_ini))
                if isnan(data_wind60_ini(i,1)) == 1
                    data_wind60_ini(i,1) = 0.5*(data_wind60_ini(i-1,1) + data_wind60_ini(i+1,1));
                end
                if isnan(data_wind60_ini(i,1)) == 1
                    data_wind60_ini(i,2) = 0.5*(data_wind60_ini(i-1,2) + data_wind60_ini(i+1,2));
                end
            end
            data = data_wind60_ini(:,2).* cosd(data_wind60_ini(:,1));
    end
    [featureset , labelset] = TimeSerisFormat(data , feature_num,move);
    Totalset = [featureset , labelset];
    Totalset(any(isnan([featureset , labelset]),2),:)= [];
    label = Totalset(:,1);
    featureset =  Totalset(:,2:end);
    clear data;