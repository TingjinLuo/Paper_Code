function fwk = weightlossfunction_theta(X,y,instance_weight,thetak, options)
% Compute the loss  function values of logistic regression

tmpxy = y.*(X*thetak);
neg_ind = find(tmpxy<0);
temp =zeros(size(tmpxy));
temp(neg_ind) = instance_weight(neg_ind).*(log(1+exp(tmpxy(neg_ind)))-tmpxy(neg_ind));
pos_ind = find(tmpxy>=0);
temp(pos_ind) = instance_weight(pos_ind).*log(1+exp(-tmpxy(pos_ind)));
% fwk = sum(temp)/size(X,1);

fwk = sum(temp)/size(X,1)+ options.lambda*thetak(2:end)'*thetak(2:end)+...
    options.lambda2*sum(thetak(2:end));