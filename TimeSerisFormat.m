function [dataset, lableset] = TimeSerisFormat(data,featurenum,move)
%TIMESEIRSFORMAT Summary of this function goes here
%   Detailed explanation goes here
%   move is the timeslot between the feature and the lable
m = max(size(data));
dataset = zeros(m-featurenum-move,featurenum);
lableset = zeros(m-featurenum-move,1);
for i=1:m-(featurenum+move)
    %should be (m-(featurenum+1)-1)
    dataset(i,:) = (data(i:i+featurenum-1,:))';
    lableset(i,:) = data(i+featurenum+move,:);
end     
end

