% implement principal component analysis
% Loads and pre-processes data

X = load('data.txt');

% center and normalize data (along columns),
% so easier to compute covariance matrx.
% bsxfun operates minus & rdivide for each row
X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide, X, std(X));

[m, n] = size(X);
% retain projection onto singular
% vectors corresponding to N largest
% singular values
N = ceil(m/2);

% Compute covariance matrix
cov = X'*X./m;

[U, S, V] = svd(cov);

% reduce dim by projecting X onto the  
% left singular vectors with the N highest
% singular values
X_red = zeros(size(X,1),N);

for i = 1:size(X,1)
    X_red(i,(1:N)) = X(i,:)*U(:,(1:N));
end

% now project back to the original space
% to obtain approximation of X using X_red

X_rec = zeros(size(X_red,1), size(U,1));

for i = 1:size(Z,1)
    X_rec(i,:) = Z(i,:)*U(:,1:K)';
end

fprintf('Approximation data from reduced data:\n');
disp(X_rec);
