function [theta_em, bag_weights, obj,theta] = WLRM(options, train_data, train_label)
% Non-Negative Logsitc Regression for Multi-Instance Learning Task 
% Update the Instance Weights of Positive bags and Extract the instances
% Ensemble Learning for undersampling training via Block Coordinate Gradient Descent
% Method onto Positive Simplex
% from Negative Bags with weights 1/ni to banlance the importance
%  Objective function: s.t.
%       [theta, bag_weights] = WLRM(options, train_data, train_label)
%
%       Input:
%           options     - A struct, includes the parameters. 
%           train_data   - An Nx1 cell,the jth instance of the ith training
%                          bag is stored in train_data{i,1}(j,:), x_j in R^d
%           train_label   - The ith label of the ith
%                           training bag is stored in  train_data{i,1}, 0
%                           is negative and 1 is positive
%
%      Output:
%           theta    - An d*1 array, the weight of each features
%           obj       - The objective function values 
%           bag_weights   - An N1x1 cell,  each cell represents the weights
%                           of each bag data in multi-instance learning.
%       Written by Tingjin Luo, Version 1.0, 01/18/2016
%       Contact: Tingjin LUO, tingjinluo@gmail.com


if nargin < 3
    fprintf('This function needs at least three input arguments.\n');
end

if ~isfield(options,'r');
    options.r = 1; % The number of positive instances constaints
end
if ~isfield(options,'Maxiter');
    options.Maxiter = 80; % The number of positive instances constaints
end
if ~isfield(options,'lambda');
    lambda = 0; % Normal Logistic Regression
else
    lambda = options.lambda; % Regularized Logistic Regression
end
if ~isfield(options,'lambda2');
    lambda2 = 0; % Normal Logistic Regression
else
    lambda2 = options.lambda2; % Regularized Logistic Regression
end
% [train_data_x,train_data_y,train_data_inx,train_data_pos,train_data_inx2] = celldatatomatrix(train_data, train_label);
Num_Bags = length(train_label); % Number of the bags
nFea = size(train_data{1,1},2); % Number of the Features
Max_Iter = options.Maxiter;

theta_em =[];
tmp_obj = [];
pos_bag_ind = find(train_label==1);
neg_bag_ind = find(train_label==0); % Negative Bags Index
Num_UnderSamples = options.num_unsmp;
% Num_UnderSamples = min(options.num_unsmp,ceil(length(neg_bag_ind)/length(pos_bag_ind)));
% Initialize the weights of each bag, i.e. bag_weights{i,1} =[1,1,..,1]/n_i
posinstance_weights =[];
for i=1:length(pos_bag_ind)
    nSmp_Bags(i) = size(train_data{pos_bag_ind(i),1},1); % The number of instances in each positve bag
    bag_weights_ini{i,1} = ones(nSmp_Bags(i),1)./nSmp_Bags(i);
    posnew_train_data(i,:) = bag_weights_ini{i,1}'*train_data{pos_bag_ind(i),1}; 
end
Pos_train_data = train_data(pos_bag_ind,1);
posnew_train_label(1:length(pos_bag_ind),1) =1;
posinstance_weights(1:length(pos_bag_ind),1) =1;
for undersmp_id=1:Num_UnderSamples
    % To solve the inbalanced classification problem by UnderSampling
    neg_bag_idx = (randperm(length(neg_bag_ind)))';
    neg_bag_idx_unsmp = neg_bag_ind(neg_bag_idx(1:length(pos_bag_ind)));
    train_data_unsmp = train_data([pos_bag_ind;neg_bag_idx_unsmp],1);
    train_label_unsmp = train_label([pos_bag_ind;neg_bag_idx_unsmp]);
    new_train_data_unsmp = [];
    new_train_label_unsmp= [];
    instance_weights_unsmp = [];
    new_train_data_unsmp = posnew_train_data;
    new_train_label_unsmp = posnew_train_label;
    instance_weights_unsmp = posinstance_weights;
    for i=1: length(neg_bag_idx_unsmp)
        new_train_data_unsmp = [new_train_data_unsmp; train_data_unsmp{i+length(pos_bag_ind),1}];
        tmp_num =size(train_data_unsmp{i+length(pos_bag_ind),1},1);
        new_train_label_unsmp = [new_train_label_unsmp; zeros(tmp_num,1)];
        instance_weights_unsmp =[instance_weights_unsmp; options.rho*ones(tmp_num,1)/tmp_num];
        % instance_weights_unsmp =[instance_weights_unsmp; ones(tmp_num,1)];
    end
   % instance_weights_unsmp(length(pos_bag_ind)+1:end) = instance_weights_unsmp(length(pos_bag_ind)+1:end)*length(pos_bag_ind)/(length(instance_weights_unsmp)-length(pos_bag_ind));
    % bag_weights = bag_weights_ini;
    
    binary_label = 2*new_train_label_unsmp-1;
    [nSmp, nFea] = size(new_train_data_unsmp); % number of examples and number of  parameters (features)
    theta1 = zeros(nFea+1,1); % (n+1) to account for the x0 term
    theta0 = zeros(nFea+1,1);
    bag_weights1 = bag_weights_ini;
    bag_weights0 = bag_weights_ini;
    para(1) = 0;
    para(2) =1;
    % Add ones to the X data matrix to account for x0
    new_train_data_unsmp = [ones(nSmp, 1) new_train_data_unsmp];
    theta_unsmp = theta1;
    bag_weights = bag_weights1;
    for iter =1: Max_Iter
        para(iter+1) =  0.5*(1+sqrt(1+4*para(iter)^2));
        tmp_gama = (1-para(iter))/para(iter+1);
        % First Step: Fixed the weights of each bag u_i, update the feature weights theta w
        % Compute the gradient of w
        grad_theta =[];
        yxwk = binary_label.*(new_train_data_unsmp*theta1);
        pos_indk = find(yxwk>=0);
        if length(pos_indk)>0
            grad_theta = -new_train_data_unsmp(pos_indk,:)'*(instance_weights_unsmp(pos_indk).*binary_label(pos_indk).*exp(-yxwk(pos_indk))./(1+exp(-yxwk(pos_indk))));
        end
        neg_indk = find(yxwk<0);
        if length(neg_indk)>0
            grad_theta = grad_theta-new_train_data_unsmp(neg_indk,:)'*(instance_weights_unsmp(neg_indk).*binary_label(neg_indk)./(1+exp(yxwk(neg_indk))));
        end
        % The second term is regularizer
        grad_theta=(1/nSmp)*grad_theta+2*lambda*theta1 +lambda2*sign(theta1);
        % backtracking line search, f(xk-alpha*f'(xk))<= f(xk)-alpha*gama*||f'(xk)||^2
        alphawk = BacktrackingLineSearch_My(@(t)weightlossfunction_theta(new_train_data_unsmp,binary_label,instance_weights_unsmp,t, options), theta1, grad_theta);
        theta_unsmp = (1-tmp_gama)*theta1+  tmp_gama*theta0 - alphawk*grad_theta;
        
        % Second Step: Fixed w, update the weights of each bag u_i by Projected Gradient Descent
        %  Compute the gradient of each bag weights
        obj(iter) =0;
        for i=1:length(pos_bag_ind)
            % Computing the gradient each bag weights u_i
            each_bag_data = [ones(size(train_data_unsmp{i,1},1),1) train_data_unsmp{i,1}];
            grad_bag_weightsi =zeros(size(each_bag_data,1),1);
            yxwk = binary_label(i)*(each_bag_data*theta1)'*bag_weights1{i,1};
            % grad_bag_weightsi = -binary_label(i)*train_data_unsmp{i,1}*theta_unsmp./(1+exp(yxwk))/size(train_data_unsmp{i,1},1);
            pos_indk = find(yxwk>=0);
            if length(pos_indk)>0
                grad_bag_weightsi(pos_indk) = -binary_label(i)*each_bag_data(pos_indk,:)*theta1.*exp(-yxwk(pos_indk))./(1+exp(-yxwk(pos_indk)));
            end
            neg_indk = find(yxwk<0);
            if length(neg_indk)>0
                grad_bag_weightsi(neg_indk) = grad_bag_weightsi(neg_indk) -binary_label(i)*each_bag_data(neg_indk,:)*theta1./(1+exp(yxwk(neg_indk)));
            end
            grad_bag_weightsi = grad_bag_weightsi/size(each_bag_data,1);
            
            % backtracking line search, f(xk-alpha*f'(xk))<= f(xk)-alpha*gama*||f'(xk)||^2
            alphauik = BacktrackingLineSearch_My(@(t)lossfunction_gu(each_bag_data,theta1,binary_label(i),t), bag_weights1{i,1}, grad_bag_weightsi);
            tmp_uik = (1-tmp_gama)*bag_weights1{i,1}+  tmp_gama*bag_weights0{i,1} - alphauik*grad_bag_weightsi;
            
            % Compute Sparse Projection onto Simplex
            bag_weights{i,1} = SparseSimplexProj(tmp_uik,options);
            % Update the training data of logistic regression
            new_train_data_unsmp(i,:) = bag_weights{i,1}'*[ones(nSmp_Bags(i),1) train_data{pos_bag_ind(i),1}];
            
            obj(iter) = obj(iter) + lossfunction_gu(each_bag_data,theta1,binary_label(i),bag_weights1{i,1});
        end
        
        neg_ind_unsmp = find(new_train_label_unsmp==0);
        tmpxy = binary_label(neg_ind_unsmp).*(new_train_data_unsmp(neg_ind_unsmp,:)*theta1);
        tmp_neg_obj = zeros(size(neg_ind_unsmp));
        neg_ind_xy = find(tmpxy<0);
        tmp_neg_obj(neg_ind_xy) = log(1+exp(tmpxy(neg_ind_xy)))-tmpxy(neg_ind_xy);
        pos_ind_xy = find(tmpxy>=0);
        tmp_neg_obj(pos_ind_xy) = log(1+exp(-tmpxy(pos_ind_xy)));
        tmp_obj_unsmp = instance_weights_unsmp(neg_ind_unsmp).*tmp_neg_obj;
   
        obj(iter) = obj(iter) + sum(tmp_obj_unsmp)+lambda*theta1'*theta1 +lambda2*sum(abs(theta1));
        
        bag_weights0 = bag_weights1;
        bag_weights1= bag_weights;
        theta0 = theta1;
        theta1 = theta_unsmp;
        
%         if abs(obj(iter)-temp) < options.epsilon
%             break;
%         end
        fprintf('Iteration \t %d| Cost: \t%d.\n', iter, obj(iter));
    end
    fprintf('The %d-th Training (Total:%d) is Comlished.\n',undersmp_id, Num_UnderSamples);
    theta_em =[theta_em theta_unsmp];
    tmp_obj = [tmp_obj;obj];
end

theta = sum(theta_em,2)./Num_UnderSamples;
obj = mean(tmp_obj);


