
function[corr,grad1,grad2]=DCCA_corr(X1,X2,K,rcov1,rcov2) 
%   computes the total 
%   correlation and gradient of total correlation with respect to the 
%   data matrices.
% 
% Inputs
%   X1/X2: network output for view 1/view 2.
%   K: dimensionality of CCA projection.
%   rcov1/rcov2: optinal regularization parameter for view 1/view 2.
% 
% Outputs
%   corr: total correlation.
%   grad1/grad2: gradient of the DCCA objective with respect to X1/X2.

if ~exist('rcov1','var') || isempty(rcov1)
  rcov1=0;
end
if ~exist('rcov2','var') || isempty(rcov2)
  rcov2=0;
end
[N,d1] =size(X1); [~,d2] =size(X2);

%%
% Remove mean.
m1=mean(X1,1); X1=X1-repmat(m1,N,1);
m2=mean(X2,1); X2=X2-repmat(m2,N,1);

S11=(X1'*X1)/(N-1)+rcov1*eye(d1); S22=(X2'*X2)/(N-1)+rcov2*eye(d2); 
S12=(X1'*X2)/(N-1);
[V1,D1]=eig(S11); [V2,D2]=eig(S22);
% For numerical stability.
D1=diag(D1); idx1=find(D1>1e-12); D1=D1(idx1); V1=V1(:,idx1);
D2=diag(D2); idx2=find(D2>1e-12); D2=D2(idx2); V2=V2(:,idx2);

K11=V1*diag(D1.^(-1/2))*V1'; K22=V2*diag(D2.^(-1/2))*V2';
T=K11*S12*K22; [U,D,V]=svd(T,0); 
U=U(:,1:K); D=D(1:K,1:K); V=V(:,1:K);
corr=sum(sum(D));

if nargout>1
  DELTA12=(K11*U)*(V'*K22);
  DELTA11=-0.5*(K11*U)*D*(U'*K11);
  DELTA22=-0.5*(K22*V)*D*(V'*K22);
  
  grad1=2*X1*DELTA11+X2*DELTA12'; grad1=grad1/(N-1);
  grad2=X1*DELTA12+2*X2*DELTA22; grad2=grad2/(N-1);
end
end
