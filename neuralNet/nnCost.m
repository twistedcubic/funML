function [C Grad]  = nnCost(X, y, Theta1, Theta2, lambda   )
% Cost function for neural net

% X is input data, y is actual output
% lambda is regularization parameter
% theta1 theta2 are current parameter values


m = size(X, 1);

% add bias/intercept term
X = [ones(m, 1), X];

rand(l2_sz, l1_sz);


% feedforward first to compute predicted output with current parameters


    % matrix of input layer activation values

    a2 = sigmoid(X * Theta1');
    %add one unit for intercept term
    a2 = [ones(size(a2,1),1) a2];

    % matrix of hidden layer activation values
    a3 = sigmoid(a2 * Theta2');

% regularization term from element wise product of
% parameter matrices, except the parameters for the 
% bias terms

r = (sum(sum(Theta1.*Theta1)) + sum(sum(Theta2.*Theta2)) - ...
    Theta1(:,1)'*Theta1(:,1) - Theta2(:,1)'*Theta2(:,1) )*lambda/2/m;

    % Logistic cost function
    C = (-y'*log(a3 ) - (1 - y)'*log(1-a3))/m + r;

% backpropagation to compute gradient of cost with respect to Theta1 and Theta2
% using output just computed
% delta's are the error terms propagated through the layers

delta3 = A3 - y;

% pull back the error to from output to hidden layer.
% This is numerical chain rule. sigmoidGrad is derivative of sigmoid function
delta2 = delta3*Theta2 .* sigmoidGrad([ones(size(z2,1),1) z2]);

% remove bias term
delta2 = delta2(:, 2:end);

% project onto actual data
delta1 = delta2' * X;

%project onto activation values in hidden layer
delta2 = delta3' * a2;

% add regularization. Again not for bias terms
Theta1Grad = (delta1 * lambda.*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)] )./m;
Theta2Grad = (delta2 * lambda.*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)] )./m;

% Unroll the gradients
Grad = [Theta1Grad(:); Theta2Grad(:)];

end