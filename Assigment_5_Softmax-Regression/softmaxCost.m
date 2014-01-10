function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%
% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);
numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.


%========================================================================
% Computing the cost function:
% First term: Fidelity term
aux1 = theta*data;
aux2 = bsxfun(@minus, aux1, max(aux1, [], 1)); %Substracts the maximm value of the matrix "aux".
aux3 = exp(aux2);
aux4 = bsxfun(@rdivide, aux3, sum(aux3)); %I divides the vector "aux3" by the sum of its elements.
aux5= groundTruth.*aux4;
aux6 = log(aux5(aux5 ~= 0)); %Extract non-zero entries.
% cost = -sum(aux5(:))/sizeof(data,2);
cost_term1 = -mean(aux6);

% Second term: weight decay
theta_vec = theta(:);
theta_vec = theta_vec.^2;
cost_term2 = (lambda/2)*sum(theta_vec);

cost = cost_term1 + cost_term2;
%========================================================================
% Computing the gradient:
from_fidelity_term = (groundTruth - aux4);
from_weight_decay = lambda.*theta;
thetagrad = ((-1/size(data,2))*data*from_fidelity_term')' + from_weight_decay; %OJOOOOOOOOOO

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end
