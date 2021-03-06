function [C, grad] = cost(X, y, theta, h)
%computes the cost function. X is data matrix, y is observed values vector
%theta is column vec of parameters. h is a function that computes expected value
%returns the cost and gradient 

m = size(X, 1);
%depends on the cases for h
%for logistic regression
C = (-y' * log(h(X*theta) ) - (1-y)'*log(1 - h(X*theta)))/m;

grad = (((h(X*theta)-y).'*X)')./m; % + lambda.*theta
% grad(1) = grad(1)-lambda*theta(1)/m;

end