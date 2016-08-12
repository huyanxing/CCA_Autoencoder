function [model, errors] = crbmBB(X, numhid,  varargin)
%Learn RBM with Bernoulli hidden and visible units
%This is not meant to be applied to image data
%code by Andrej Karpathy
%based on implementation of Kevin Swersky and Ruslan Salakhutdinov

%INPUTS: 
%X              ... data. should be binary, or in [0,1] to be interpreted 
%               ... as probabilities
%numhid         ... number of hidden layers

%additional inputs (specified as name value pairs or in struct)
%method         ... CD or SML 
%eta            ... learning rate
%momentum       ... momentum for smoothness amd to prevent overfitting
%               ... NOTE: momentum is not recommended with SML
%maxepoch       ... # of epochs: each is a full pass through train data
%sig_high_asy   ... the high asymtotes of the sigmoid function 
%sig_low_asy    ... the low asymtotes of the sigmoid function 
%delta          ... the trade off of the noise control
%avglast        ... how many epochs before maxepoch to start averaging
%               ... before. Procedure suggested for faster convergence by
%               ... Kevin Swersky in his MSc thesis
%penalty        ... weight decay factor
%batchsize      ... The number of training instances per batch
%verbose        ... For printing progress
%anneal         ... Flag. If set true, the penalty is annealed linearly
%               ... through epochs to 10% of its original value

%OUTPUTS:
%model.type     ... Type of RBM (i.e. type of its visible and hidden units)
%model.W        ... The weights of the connections
%model.b        ... The biases of the hidden layer
%model.c        ... The biases of the visible layer
%model.top      ... The activity of the top layer, to be used when training
%               ... DBN's
%errors         ... The errors in reconstruction at every epoch

%Process options
%if args are just passed through in calls they become cells
if (isstruct(varargin)) 
    args= prepareArgs(varargin{1});
else
    args= prepareArgs(varargin);
end
[   
    eta_W          ...
    eta_a         ...
    momentum      ...
    maxepoch      ...
    sig_high_asy  ...
    sig_low_asy   ...
    delta         ... 
    avglast       ...
    penalty       ...
    batchsize     ...
    verbose       ...
    anneal        ...
    ] = process_options(args    , ...
    'eta_W'         ,  0.0001      , ...
    'eta_a'         ,  0.0001     , ...
    'momentum'      ,  0      , ...
    'maxepoch'      ,  10000       , ...
    'sig_high_asy'  ,  1        , ...
    'sig_low_asy'   ,  -1        , ...
    'delta'         ,  0.0001      , ...
    'avglast'       ,  5        , ...
    'penalty'       , 0      , ...
    'batchsize'     , 100       , ...
    'verbose'       , false     , ...
    'anneal'        , false);
avgstart = maxepoch - avglast;
oldpenalty= penalty;
[N,d]=size(X);

if (verbose) 
    fprintf('Preprocessing data...\n');
end
%% processing data
%Create batches
numcases=N;
numdims=d;
numbatches= ceil(N/batchsize);
groups= repmat(1:numbatches, 1, batchsize);
groups= groups(1:N);
perm=randperm(N);
for  i=1:numbatches
     batchdata{i}= X(groups==i,:);
end

%% initialization 
%train RBM
W = 0.1*randn(numdims,numhid);
ac = rand(1,numdims); % nosie-control in visible nodes
ab = rand(1,numhid);  % nosie-control in hidden nodes
ph = zeros(numcases,numhid); %positive hidden nodes
nh = zeros(numcases,numhid); %negitive hidden nodes
phstates = zeros(numcases,numhid); %positive hidden stastes
nhstates = zeros(numcases,numhid); %negtive hidden stastes 
negdata = zeros(numcases,numdims); %negtive visible nodes
negdatastates = zeros(numcases,numdims);%negtive visible states
Winc  = zeros(numdims,numhid);
abinc = zeros(1,numhid);
acinc = zeros(1,numdims);
Wavg = W;
abavg = ab;
acavg = ac;
t = 1;
errors=zeros(1,maxepoch);

for epoch = 1:maxepoch
    
	errsum=0;
    if (anneal)
        %apply linear weight penalty decay
        penalty= oldpenalty - 0.9*epoch/maxepoch*oldpenalty;
    end
    
    for batch = 1:numbatches
		[numcases numdims]=size(batchdata{batch});
		data = batchdata{batch};
        
        %% go up
        noise_hid = randn(1,numhid);
        input_up = data*W+delta*repmat(noise_hid,numcases,1);
        ph = logistic(repmat(ab,numcases,1).*input_up);%input to the hidden nodes in the positive phase
		phstates = sig_low_asy+((sig_high_asy-sig_low_asy).*ph);
        
		%phstates = ph > rand(numcases,numhid);
%         if (isequal(method,'SML'))
%             if (epoch == 1 && batch == 1)
%                 nhstates = phstates;
%             end
%         elseif (isequal(method,'CD'))
%             nhstates = phstates;
%         end
		
        %% go down
        noise_vis =randn(1,numdims);
        nhstates=phstates;
        input_down = nhstates*W'+delta*repmat(noise_vis,numcases,1);
        negdata = logistic(repmat(ac,numcases,1).*input_down);
		negdatastates = sig_low_asy+((sig_high_asy-sig_low_asy).*negdata);
        
        %% go up one more time
		noise_hid = randn(1,numhid);
        input_up = negdatastates*W+delta*repmat(noise_hid,numcases,1);
        ph = logistic(repmat(ab,numcases,1).*input_up);%input to the hidden nodes in the positive phase
		phstates = sig_low_asy+((sig_high_asy-sig_low_asy).*ph);
        
		
        %update weights and biases
        dW = (data'*phstates - negdatastates'*nhstates);
        dac = (sum(data.^2) - sum(negdatastates.^2))./(ac.^2);
        dab = (sum(phstates.^2) - sum(nhstates.^2))./(ab.^2);
        
       
		Winc = momentum*Winc + eta_W*(dW/numcases - penalty*W);
		abinc = momentum*abinc + eta_a*(dab/numcases);
		acinc = momentum*acinc + eta_a*(dac/numcases);
		W = W + Winc;
		ab = ab + abinc;
		ac = ac + acinc;
        
        if (epoch > avgstart)
            %apply averaging
			Wavg = Wavg - (1/t)*(Wavg - W);
			acavg = acavg - (1/t)*(acavg - ac);
			abavg = abavg - (1/t)*(abavg - ab);
			t = t+1;
		else
			Wavg = W;
			abavg = ab;
			acavg = ac;
        end
        
        %accumulate reconstruction error
        err= sum(sum( (data-negdatastates).^2 ));
		errsum = err + errsum;
    end
    
    errors(epoch)=errsum;
    if (verbose) 
        fprintf('Ended epoch %i/%i. Reconstruction error is %f\n', ...
            epoch, maxepoch, errsum);
    end
end

model.type= 'CBB';
%model.top= sig_low_asy+(sig_high_asy-sig_low_asy).*(logistic(repmat(abavg,N,1).*(X*W+repmat(delta*noise_hid,N,1))));
model.W= Wavg;
model.ab= abavg;
model.ac= acavg;

