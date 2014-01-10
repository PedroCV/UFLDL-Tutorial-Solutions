function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));

%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%

% Forward pass:
in2 = stack{1}.w*data + repmat(stack{1}.b,1,M);
sal2 = 1./(1+exp(-in2));
in3 = stack{2}.w*sal2 + repmat(stack{2}.b,1,M);
sal3 = 1./(1+exp(-in3));
in4 = softmaxTheta*sal3;
aux2 = bsxfun(@minus, in4, max(in4, [], 1)); %Substracts the maximm value of the matrix "aux".
aux3 = exp(aux2);
aux4 = bsxfun(@rdivide, aux3, sum(aux3)); %It divides the vector "aux3" by the sum of its elements.
aux5= groundTruth.*aux4;
aux6 = log(aux5(aux5 ~= 0)); %Extract non-zero entries.
%== First term of cost function: fidelity term
cost_term_1 = -mean(aux6);
%== Second term of cost function: weight decay
cost_term_2 = (lambda/2)*sum(softmaxTheta(:).^2);
%= Total cost function:
cost = cost_term_1 + cost_term_2;
%========================================================================
% Computing the gradient:
from_fidelity_term = (groundTruth - aux4);
from_weight_decay = lambda.*softmaxTheta;
softmaxThetaGrad = ((-1/M)*sal3*from_fidelity_term')' + from_weight_decay;
%
deriv_3 = ((-1/M)*(softmaxTheta'*from_fidelity_term)).*(sal3.*(1-sal3));
stackgrad{2}.w = deriv_3*sal2';
% stackgrad{2}.b = deriv_3*ones(size(sal2,2),1);
stackgrad{2}.b = deriv_3*ones(M,1);
%
deriv_2 = ((stack{2}.w')*deriv_3).*(sal2.*(1-sal2));
stackgrad{1}.w = deriv_2*data';
% stackgrad{1}.b = deriv_2*ones(size(sal1,2),1);
stackgrad{1}.b = deriv_2*ones(M,1);
% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
