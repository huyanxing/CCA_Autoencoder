function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));
%%
thetaMatrix = repmat(theta,1,size(theta));
ep =  1.0000e-04;
diffmatrix=ep*eye(size(theta,1));

for i=1: size(theta,1)
    [temp_1,~]=J((thetaMatrix(:,i)+diffmatrix(:,i)));
     [temp_2,~]=J((thetaMatrix(:,i)-diffmatrix(:,i)));
    numgrad(i)=(temp_1-temp_2)/(2.0*ep);
    fprintf('loop %d \n ',i)
end
%% ---------------------------------------------------------------
 end
