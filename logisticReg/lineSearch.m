function alpha = lineSearch(theta0, cost_fn, p)
% sets step size using line search
% x0 is current location
% p is step direction

alpha = 0.1;
% base tuning parameter
gamma = 0.7;
% theta after step
theta = theta0 + p'*alpha;

[C0, G] = cost_fn(theta);
%directional derivative


while (C0 + alpha * (G'*p) < C)
    alpha = alpha * alpha;
    theta = theta0 + p'*alpha;  
    [C, G] = cost_fn(theta)
end