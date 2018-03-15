function X = SimplexProj(Y)
% Projection Onto Positive Simplex
% Y = [y1,y2...yN]
% X = [x1,x2...xN]
Y = Y';
[N,D] = size(Y);
X = sort(Y,2,'descend');
Xtmp = (cumsum(X,2)-1)*diag(sparse(1./(1:D)));
X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);
X = X';