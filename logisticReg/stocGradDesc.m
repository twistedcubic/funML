function [theta, C] = stocGradDesc(X, cost_fn, theta0, alpha, maxIter)
% computes theta that minimizes cost function h, using
% initial parameter theta0, up to maxIter iterations over all samples.
% Uses stochastic gradient descent

[m, n] = size(X);

%use pick_sz training samples instead of 1 at a time for smoother convergence
pick_sz = 5;
theta_vec = zeros(n, ceil(m/pick_sz));
cost_vec = zeros(ceil(m/pick_sz),1);
theta_vec(:, 1) = theta0;

for j = 1:maxIter
    sz = pick_sz;
    for i = sz+1:sz:m
        if i+sz-1 > m
            sz = m - i;
        end
        X_temp = X(i:(i+sz-1), :);
        [cost, G] = cost_fn(X_temp, theta_vec(:, floor(i/sz)));
        theta_vec(:, floor(i/sz)+1) = theta_vec(:, floor(i/sz)) - G.*alpha;
        cost_vec(floor(i/sz)+1) = cost;

    end
end
theta = theta_vec(ceil(m/pick_sz));
C = cost_vec(ceil(m/pick_sz));

end