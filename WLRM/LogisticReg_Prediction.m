function [pred_label,pred_score , accuracy] = LogisticReg_Prediction(theta, test_data, test_label)
% Logsitc Regression Classifier Prediction for Multi-Instance Learning Task
%       [pred_label] = LogisticReg_Prediction(test_data, test_label)
%
%       Input:
%           test_data   - An Nx1 cell,the jth instance of the ith training
%                          bag is stored in test_data{i,1}(j,:), x_j in R^d
%           test_label   - The ith label of the ith
%                           training bag is stored in  test_data{i,1} 
%
%      Output:
%           pred_label    - An N*1 vector, the predicted label by Logistic
%           Regression Classifier
%       Written by Tingjin Luo, Version 1.0
%       Contact: Tingjin LUO


Num_Bags = length(test_label); % Number of the bags
[nFea, Num_UnSmp] = size(theta);
if Num_UnSmp ==1
	for i=1:Num_Bags
    	[pred_label(i,1),pred_score(i,1)] =  predict(theta, test_data{i,1});
	end
else
	for j =1:Num_UnSmp
		for i=1:Num_Bags
    		[pred_label_unsmp(i,j), pred_score_unsmp(i,j)] =  predict(theta(:,j), test_data{i,1});
		end
	end
	% Majority Voting for multiple Classifiers
	[pred_label,~] = majorityvoting(pred_label_unsmp');
    pred_label = pred_label';
	pred_score = max(pred_score_unsmp,[],2);
end

accuracy = length(find(pred_label== test_label))/length(test_label);


function [pred_bag, score_bag, prediction] = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
% X = mxn matrix
% theta = (n+1)x1 column vector

m = size(X, 1); % Number of training examples
prediction = zeros(size(X, 1), 1);

X = [ones(m, 1) X];
prediction = (sigmoid(X*theta) >= 0.5);
% prediction = ((X*theta) >= 0);

score_bag = sigmoid(max(X*theta));
% score_bag = mean(X*theta);

ind = find(prediction ==1);

if length(ind) >0
    pred_bag = 1;
else 
    pred_bag = 0;
end