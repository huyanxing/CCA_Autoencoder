function [ Trainset, Testset ] = NCrossPart ( totalset,N)
%N-CrossPart used to partion the samplesets for N-cross validation 
%Input- totalset is an m-n matrix where m is the number of samples and n is
%the number of attributes
%N-is a positive integer which is the parameter of the N-Cross, default as 10
%when N is set as equal to m, this is for leave-one-out
%Output-Trainset and Testset are cells
%Trainset{i} corresponds to Testset{i}


%%  Defualt input
if nargin<2
    N=10;
end 

%% Initialze variables
[sample_num,~] = size(totalset);
Index_vec = (1:sample_num);
adjust_temp = rem(sample_num,N);
adjust = zeros(1,N);
adjust(1:adjust_temp) = 1;
segment_length=floor(sample_num/N);
m=sample_num;

%% Get the index of test set

for i=1:N
    index_temp = randperm(m,segment_length+adjust(i));
    testset_index{i}=Index_vec([index_temp]);
    Index_vec([index_temp]) = [ ];
    m=m-(segment_length+adjust(i));
end

%% Date divsion
for i=1:N
    temp=totalset;
    Testset{i}=temp([testset_index{i}],:); 
    temp([testset_index{i}],:)=[ ];
    Trainset{i}=temp;
end

end

