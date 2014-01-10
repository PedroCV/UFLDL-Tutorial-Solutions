function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
aux1 = theta*data;
aux2 = bsxfun(@minus, aux1, max(aux1, [], 1)); %Substracts the maximm value of the matrix "aux".
aux3 = exp(aux2);
aux4 = bsxfun(@rdivide, aux3, sum(aux3)); %I divides the vector "aux3" by the sum of its elements.

for i=1:size(data,2)
    aux = aux4(:,i);
    pred(i) = find(aux == max(aux));
end
% aux5= groundTruth.*aux4;
% aux6 = log(aux5(aux5 ~= 0)); %Extract non-zero entries.
% cost_term1 = -mean(aux6);
