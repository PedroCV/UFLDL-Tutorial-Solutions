function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
M = size(data,2);
in2 = stack{1}.w*data + repmat(stack{1}.b,1,M);
sal2 = 1./(1+exp(-in2));
in3 = stack{2}.w*sal2 + repmat(stack{2}.b,1,M);
sal3 = 1./(1+exp(-in3));
in4 = softmaxTheta*sal3;
aux2 = bsxfun(@minus, in4, max(in4, [], 1)); %Substracts the maximm value of the matrix "aux".
aux3 = exp(aux2);
aux4 = bsxfun(@rdivide, aux3, sum(aux3)); %It divides the vector "aux3" by the sum of its elements.

pred = zeros(size(data,2),1);
for i=1:size(data,2)
    aux = aux4(:,i);
    pred(i) = find(aux == max(aux));
end
% aux5= groundTruth.*aux4;
% aux6 = log(aux5(aux5 ~= 0)); %Extract non-zero entries.

