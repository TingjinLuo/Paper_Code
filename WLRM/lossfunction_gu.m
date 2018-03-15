function fuk = lossfunction_gu(X,theta,y,uk)
% Compute the loss of each u_i

tmpxy = y*(X*theta)'*uk;
if tmpxy>0
    fuk = log(1+exp(-tmpxy));
else
    fuk = log(1+exp(tmpxy))-tmpxy;
end