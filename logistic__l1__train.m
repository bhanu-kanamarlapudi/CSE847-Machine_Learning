data = importdata('ad_data.mat');
features = importdata('feature_name.mat');
train_X = data.X_train;
train_y = data.y_train;
test_X = data.X_test;
test_y = data.y_test;
param = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
fprintf("Num of positive samples - %d\n",sum(train_y==1));
fprintf("Num of negative samples - %d\n",sum(train_y==-1));
fprintf("Percentage of positive samples - %f\n",sum(train_y==1)/size(train_y,1));
fprintf("Percentage of negative samples - %f\n",sum(train_y==-1)/size(train_y,1));

for i=1:size(param,2)
    [w,c] = logistic_l1_train(train_X, train_y, param(i));
    predict_y = sigmoid(test_X* w + c);
    [X, Y, T, AUC] = perfcurve(test_y,predict_y,1);
    num_features_selected= sum(w~=0);
    fprintf("Par - %f, AUC - %f, Number of features Selected - %d \n", param(i), AUC, num_features_selected);
end

function [w,c] = logistic_l1_train(data, labels, par)
% OUTPUT w is equivalent to the first d dimension of weights in logistic train
% c is the bias term, equivalent to the last dimension in weights in logistic train.
% Specify the options (use without modification).
    opts.rFlag = 1; % range of par within [0, 1].
    opts.tol = 1e-6; % optimization precision
    opts.tFlag = 4; % termination options.
    opts.maxIter = 5000; % maximum iterations.
    %fprintf("%f",par);
    [w, c] = LogisticR(data, labels, par, opts);
end

function [res] = sigmoid(z)
    res = 1./(1+exp(-z));
end