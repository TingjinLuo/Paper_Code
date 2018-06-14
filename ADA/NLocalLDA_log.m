function [W, S, obj] = NLocalLDA_log(X, gnd, d, options,expa)
%New Local LDA for supervised dimensionality reduction
% max_{W,S}  \sum_{i=1}^{c}\sum_{j=1}^{ni} \sum_{k=1}^{ni}  s_ij ||W'*(xji
% - xki)||^2, s.t.  W' St W=I
% [W, S, obj] = NLocalLDA(X, gnd, d, options);
% Input:
%       X  -- data matrix, each row vector of data is a data point.
%       d  -- projected dimension
%       gnd  --  Colunm vector of the label information for each data point.
%       options  -- Struct value in Matlab. The fields in options
%                          that can be set:
%              IterMax  -- The repeat times of the
%                     iterative procedure. Default 50
%              t  -- The parameter needed under 'HeatKernel'
%                          WeightMode. Default will be 1
%              epsilon  -- The precision of iteration
% Output:
%       W-- dim*d projection matrix
%       S --  n*n learned symmetric similarity matrix
%       obj --  The objective function value
%
% Reference:
% @inproceedings{nie2014clustering,
% 	title={Clustering and projected clustering with adaptive neighbors},
% 	author={Nie, Feiping and Wang, Xiaoqian and Huang, Heng},
% 	booktitle={Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining},
% 	pages={977--986},
% 	year={2014},
% 	organization={ACM}
% }
% version 1.0 03/05/2016
% Written by Tingjin Luo (Email: tingjinluo@gmail.com)

if nargin <3
    sprintf('This function needs at least three input parameters.\n');
end

if (~exist('options','var'))
   options = [];
else
   if ~isstruct(options) 
       error('parameter error!');
   end
end

if ~isfield(options,'IterMax')
    IterMax = 50;
else
    IterMax = options.IterMax; %
end

if ~isfield(options,'t')
    nSmp = size(X,1);
    if nSmp > 3000
        D = EuDist2(X(randsample(nSmp,3000),:));
    else
        D = EuDist2(X);
    end
    options.t = mean(mean(D));
end

[nSmp, nFea] = size(X);
X = X';
%% Initialize the paramaters W and S
H = eye(nSmp)-1/nSmp*ones(nSmp);
St = X*H*X' + eps*eye(nFea);  % Scatter matrix
invSt = inv(St);

% Intialize S0
S = ones(nSmp);

opt = [];
opt.gnd = gnd;
opt.NeighborMode = 'Supervised';
opt.WeightMode = 'HeatKernel';
opt.bNormalized = 1;
opt.t = options.t;
% S = constructW(X',opt);
% S = construct_LLDA(X, opt);

%% Main Step
for iter =1:IterMax
    % Step 1: Fixed S, update W
    S0 = (S+S')/2;
    D0 = diag(sum(S0));
    L0 = D0 - S0;
    
    % SVD
    M = invSt*(X*L0*X');
    W = eig1(M, d, 0, 1);
    W = W*diag(1./sqrt(diag(W'*W)));
    
    % Compute the objective function value
    obj(iter) = trace(W'*(X*L0*X')*W);
    
    % Step 2: Fixed W, update S
    S = constructW_a(X'*W,opt,expa);
 
    
   fprintf('Iteration \t %d Finished.\n', iter);
end
tmpNorm = sqrt(sum((W').*W',2));
W = W./repmat(tmpNorm',size(W,1),1);



