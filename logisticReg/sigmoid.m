function [S] = sigmoid(X)
% sigmoid function, X is data vector, theta
% is parameter column vector

S = 1./(1 + exp(-X));

end