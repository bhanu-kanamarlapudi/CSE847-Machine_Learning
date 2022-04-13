data_x = importdata('input.xlsx');
data_y = importdata('label.xlsx');
data_y(data_y==0) = -1;
data_x(:,size(data_x,2)+1) = ones(1,size(data_x,1)); 
num_samples = size(data_x,1);
num_features = size(data_x,2);
train_size = [200, 500, 800, 1000, 1500, 2000];
accuracy = zeros(1,size(train_size,2));
test_size = 2601;
test_x = data_x(num_samples - test_size +1:num_samples,:);
test_y = data_y(num_samples - test_size +1:num_samples,:);
epsilon = 1e-5;
maxiter=1000;
for i = 1:size(train_size,2)
    train_x = data_x(1:train_size(i),:);
    train_y = data_y(1:train_size(i),:);
    weights = logistic_train(train_x ,train_y ,epsilon ,maxiter);
    predict_y = sigmoid(test_x*weights);
    predict_y(predict_y > 0.5) = 1;
    predict_y(predict_y <= 0.5) = -1;
    correct_predictions = sum(predict_y == test_y);
    acc = correct_predictions/size(test_x,1);
    accuracy(1,i) = acc;
    fprintf("Train size - %d, Accuracy - %f, Misclassification = %f \n", train_size(i),acc,1-acc);
end
figure;
hold on;
plot(train_size,accuracy);
legend('Accuracy');
hold off;

function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix withn samples and d features, where
% column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"
%
    num_features = size(data, 2);
    w = zeros(num_features, 1);
    learning_rate = 0.0001;
    i = 1;
    curr_error = -Inf;
    prev_error = Inf;
    while(i<=maxiter && abs(prev_error-curr_error)>= epsilon)
        z = labels.*(data*w);
        prev_error = curr_error;
        curr_error = mean(log(sigmoid(z)));
        dw = -(mean(sigmoid(-z).*(data.*labels)))';
        w = w - (learning_rate*dw);
        i = i+1;
    end
    weights = w;
end

function [res] = sigmoid(z)
    res = 1./(1+exp(-z));
end   
