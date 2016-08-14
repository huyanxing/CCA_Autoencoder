function [cost,grad] = SplitSparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data,subFeatureNum,gamma,K)
%%                                                         
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% visibleSize: the number of input units 
% hiddenSize: the number of hidden units  
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: training sample. 
% subfeaturenum : To show the number of features in each data matrix
% K: parameter for CCA calculation 
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W1 = reshape(W1, hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
W2 = reshape(W2, visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
%% split data
ind = 0;
for i=1:length(subFeatureNum)
    eval(['W2_', num2str(i) ,'= W2(ind+1:ind+subFeatureNum(',num2str(i),'),:);']);
    eval(['b2_', num2str(i) ,'= b2(ind+1:ind+subFeatureNum(',num2str(i),'),:);']);
    ind = ind + subFeatureNum(i);
end 

ind = 0;
for i=1:length(subFeatureNum)
    eval(['subdata', num2str(i) ,'= data(ind+1:ind+subFeatureNum(',num2str(i),'),:);']);
    ind = ind + subFeatureNum(i);
end 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% --------disp([numgrad grad]); 

sample_num=max(size(data));
%calculate the 2nd layer
z2 = W1*data+repmat(b1,1,sample_num);
a2= sigmoid(z2);
%calculate the 3 layer 
ind = 0;
for ind = 1:length(subFeatureNum)
    eval(['z3_',num2str(ind),' = W2_',num2str(ind),'*a2 + repmat(b2_',num2str(ind),',1,sample_num);']);
    eval(['suba3_',num2str(ind),'= sigmoid(z3_',num2str(ind),');']);
end 
%% splite the output
cost_main = 0;

% error from the data : least squre 
for i = 1:length(subFeatureNum)
    subError = 0;
    eval(['subError = sum(sum((subdata',num2str(i),'-suba3_',num2str(i),').^2));']);
    cost_main = cost_main + subError;
end
cost_main = (0.5/sample_num)*cost_main;

% regularization
weight_decay = 0.5*(sum(sum(W1.^2))+sum(sum(W2.^2)));%the weigh dacay
rho = (1/sample_num)*sum(a2,2);
Regterm =  sum(sparsityParam.*log(sparsityParam./rho)+(1-sparsityParam).*log((1-sparsityParam)./(1-rho)));%Sparse regularization term

%error=least squre+regularization 
cost_main =cost_main +lambda*weight_decay+beta*Regterm;

%CCA between every 2 data vectors:

%******** adjust by different conditions***********
[corr_12,grad_corr12_1,grad_corr12_2]=DCCA_corr(suba3_1',suba3_2',K);
grad_corr12_1 = grad_corr12_1';
grad_corr12_2 = grad_corr12_2';
[corr_13,grad_corr13_1,grad_corr13_3]=DCCA_corr(suba3_1',suba3_3',K);
grad_corr13_1 = grad_corr13_1';
grad_corr13_3 = grad_corr13_3';
% Total error = least squre + regularization + CCA
cost_main= cost_main-gamma*(corr_12+corr_13);
cost=cost_main;
%****** finish adjusting ************************** 
%% *********start backpropagation for grad************
% errorterm in layer3 from data L-2 norm
for i = 1:length(subFeatureNum)
    eval(['errorterm_data_',num2str(i),' = -(subdata',num2str(i),'-suba3_',num2str(i),').*sigmoidInv(z3_',num2str(i),');']),
end 
%errorterm in layer 3 from correalation  
%******** adjust by different conditions***********
errorterm_corr_1= (grad_corr12_1+grad_corr13_1).*sigmoidInv(z3_1);
errorterm_corr_2= (grad_corr12_2).*sigmoidInv(z3_2);
errorterm_corr_3= (grad_corr13_3).*sigmoidInv(z3_3);
%errorterm all 
%******** adjust by different conditions***********
errorterm_3_1 = errorterm_data_1-gamma*sample_num*errorterm_corr_1;
errorterm_3_2 = errorterm_data_2-gamma*sample_num*errorterm_corr_2;
errorterm_3_3 = errorterm_data_3-gamma*sample_num*errorterm_corr_3;

errorterm_3 = [];

for i = 1:length(subFeatureNum)
    eval(['errorterm_3 = [errorterm_3; errorterm_3_',num2str(i),'];']);
end 

% Sparse term
reg_grad =beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));
errorterm_2=(W2'*errorterm_3+repmat(reg_grad,1,sample_num)).*sigmoidInv(z2);

%% grad 
W1grad = W1grad+errorterm_2*data';
W1grad = (1/sample_num).*W1grad+lambda*W1;

gradW2Update = [];
for i = 1:length(subFeatureNum)
    eval(['gradW2Update = [gradW2Update; errorterm_3_',num2str(i),'*a2''];']);
end 
W2grad = W2grad + gradW2Update;
W2grad = (1/sample_num).*W2grad+lambda*W2;
b1grad = b1grad+sum(errorterm_2,2);
b1grad = (1/sample_num)*b1grad;
b2grad = b2grad+sum(errorterm_3,2);
b2grad = (1/sample_num)*b2grad;

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
   sigm = 1 ./ (1 + exp(-x));
end
function sigmInv = sigmoidInv(x)
    sigmInv = sigmoid(x).*(1-sigmoid(x));
end

