function x = conjGrad(x0, H, epsilon)
% uses conjugate gradient
% epsilon is tolerance
% x0

r = b - H*x0;
p = r;
alpha = (r' * r)./(p'*H*p);
x = x0 + alpha * p;
r = r - alpha * H * p';

while r > epsilon
    s = (r'*r) / (r'*r);
    p = r + s*p';    
    alpha = (r' * r)./(p'*H*p);
    x = x + alpha * p;
    r = r - alpha * H * p';
    
end

end