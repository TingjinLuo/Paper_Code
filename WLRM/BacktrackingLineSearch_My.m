function alpha = BacktrackingLineSearch_My(f, xk, grad_xk)
% Backtracking Line Search
%  alpha = BacktrackingLineSearch_My(@(t)(costfuction_regularization(t,X,y)), xk, grad)
%       Input:
%           f   - the loss function of x
%           xk    - the current point
%           grad_xk    - the gradient of current point xk
%
%      Output:
%           alpha       - the optimal step size
%       Written by Tingjin Luo, Version 1.0, 12/21/2015
%       Contact: Tingjin LUO


alpha = 1;
lambda = 0.6;
beta  = 0.9;

while f(xk-alpha*grad_xk) > f(xk) - alpha*lambda*(grad_xk'*grad_xk)
    alpha = alpha * beta;
end

