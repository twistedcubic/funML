
% K means algorithm for clustering: 
% Initialize centroids
% Loop over: 1) classify all data to nearest centroids
% 2) Update centroids


% load data
X = load('data.txt');

% number of centroids
cent_num = 5;
% number of iterations
maxIter = 10;

% randomly pick initial centroids
[m n] = size(X);
% pick cent_num random samples from X
% cent_num number of random ints 
% (not necessarily distinct) from 1 to m
randInd = randi([1 m], 1, cent_num); 
cent = zeros(cent_num, n);

for i = 1:cent_num
    cent(i) = X(randInd(i),:);
end

for i = 1:maxIter
% returns index of centroid each data point belongs to
index = findCent(X, cent);

cent = updateCent(X, index, cent_num);
end

fprintf('Centroids after %d iterations: \n', maxIter);
disp(cent);

