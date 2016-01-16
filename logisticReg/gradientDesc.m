function [theta, C] = gradientDesc(X, cost_fn, theta0, alpha, maxIter)
% computes theta that minimizes cost function h, using
% initial parameter theta0, up to maxIter iterations
% Using batch gradient descent

% record theta for debugging: 
theta_vec = zeros(size(X, 2), maxIter); % each column vec is one theta
[cost0, G]  = cost_fn(theta0);
theta_vec(:,1) = theta0;

% keep track of cost 
cost_vec = zeros(maxIter, 1);
cost_vec(1) = cost0;

for i = 2:maxIter
    [cost, G] = cost_fn(theta_vec(:, i-1));
    theta_vec(:, i) = theta_vec(:, i-1) - G.*alpha;
    cost_vec(i) = cost;
    
end

theta = theta_vec(maxIter);
C = cost_vec(maxIter);

end