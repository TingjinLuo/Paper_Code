function [train_index, test_index] = splitdata_random(label, Ratio)
% Split the data into training and testing set based on ratio
% Input:
%         label -- the label of each instance, Nx1
%         Ratio -- Ratio of Training (Percentage of training set)
% Output:
%         train_index -- The index of training set, N1x1
%         test_index --   The index of testing set,N2x1
%
%       Written by Tingjin Luo, Version 1.0, 02/17/2016
%       Contact: Tingjin LUO, tingjinluo@gmail.com


Num_Kfolds = 20;
cv_index_eachgold = crossvalind('Kfold',label,Num_Kfolds);
max_fold_id = Ratio*Num_Kfolds;
test_index = find(cv_index_eachgold > max_fold_id);
train_index = find(cv_index_eachgold <=max_fold_id);

