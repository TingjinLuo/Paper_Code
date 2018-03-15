function Y = SparseSimplexProj(weights,options)
% Sparse Projection Onto Positive Simplex

[sw, ind] = sort(weights,'descend');
if options.r < length(weights)
    sw(options.r+1:end) = 0;
end
weights = sw(ind);

% Projection on the Positive Simplex
Y = SimplexProj(weights);