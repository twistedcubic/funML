function [index] = findCent(X, cent)
% classify data to nearest centroids
% X is dataset, cent contains coordinates
% of the centroids
% returns centroid index for each sample in X

m = size(X,1);
num_cent = size(cent, 1);
index = zeros(m,1);

for i=1:m
   min = (X(i,:)-cent(1,:))*(X(i,:)-cent(1,:))';
   ind = 1;
   for j=2:num_cent
       dist = (X(i,:)-cent(j,:))*(X(i,:)-cent(j,:))';
       if dist < min
           ind = j;
           min = dist;
       end
   end
   index(i) = ind;
end

end