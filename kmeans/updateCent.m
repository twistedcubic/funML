function [cent] = updateCent(X, index, cent_num)
% compute new centroids using cent_init,
% by taking mean of samples assigned to
% each centroid
% index is array of centroid assignments
% for samples in X

[m n] = size(X);
cent = size(cent_num, n);

for i=1:cent_num
    if sum(index == i) ~= 0
        s = (idx == i)'*X;
        center = s./sum(idx==i); 
        cent(i,:) = center';
    end
end

end