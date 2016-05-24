function [activation] = getOptFeatures(W,b, data)
%W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
%b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
sample_num = max(size(data));
input_1 = W*data+repmat(b,1,sample_num);
output_1 = sigmoid(input_1);
%-------------------------------------------------------------------
activation = output_1;
end
function sigm = sigmoid(x)
   sigm = 1 ./ (1 + exp(-x));
end
function sigmInv = sigmoidInv(x)
    sigmInv = sigmoid(x).*(1-sigmoid(x));
end