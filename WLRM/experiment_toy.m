% function experiment_toy()
% Experiment on the systhetic data

% clear all;
% clc

% generate two classes from two Gaussian function
nSmp = 30;
num_pos = 10;
nFea  =  2;
delta1 = 0.75;
delta2 = 0.75;
mu = 1;
mu1 = mu*ones(1,nFea);
mu2 = -mu*ones(1,nFea);
sigma1 = diag(ones(1,nFea)*delta1);
sigma2 = diag(ones(1,nFea)*delta2);

gnd_inst =[];
inst_data = [];
for id=1: nSmp
    numinst = randi([1 10],1);
    if id <=num_pos
        % Positive Bag
        data{id,1} = [mvnrnd(mu1,sigma1,1); mvnrnd(mu2,sigma2,numinst-1)];
        label(id,1) = 1;
        gnd_inst = [gnd_inst; [1; -ones(numinst-1,1)]];
    else
        data{id,1} = mvnrnd(mu2,sigma2,numinst);
        label(id,1) = -1;
        gnd_inst = [gnd_inst; label(id,1)*ones(numinst,1)];
    end
    inst_data = [inst_data;data{id,1}];
    % gnd_inst = [gnd_inst; label(id,1)*ones(numinst,1)];
end

Num_UnSmp = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the Parameters
options = [];
options.r = 1;
options.Maxiter = 50;
options.lambda = 0.001;
options.lambda2 = 0.001;
options.epsilon = 1e-6;
options.rho = 2;
options.num_unsmp = Num_UnSmp;

each_exp_label_binary = (label +1)./2;
train_index = [1:nSmp];
ttrain_data = data(train_index,:);
ttrain_label = each_exp_label_binary(train_index);

% WLRMI
[theta_wlrmi, bag_weights_wlrmi, obj] = WLRM(options, ttrain_data, ttrain_label);
