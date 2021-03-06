% imports and pre-processes transition  matrix data 
% and samples
% Compute optimal parameters for transition matrices

% set the layer sizes
% input layer
l1_sz = 20;
% hidden layer
l2_sz = 30;
% output layer
l3_sz = 10;

% Initialize parameters to random values to break symmetry
% (for faster convergence)
theta1_init = rand(l2_sz, l1_sz);
theta2_init = rand(l3_sz, l2_sz);

% Train our neural net using the cost function and gradient
% using gradient descent

% regularization parameter
lambda = 0.1;

[C, G] = gradDesc2(X, y, @(theta1, theta2)cost_fn(X, y, theta1, theta2, lambda), theta1_init, theta2_init);

fprintf('cost with one round of backpropagation: %d\n',C);
fprintf('gradient with backpropogation:\n');
disp(G);