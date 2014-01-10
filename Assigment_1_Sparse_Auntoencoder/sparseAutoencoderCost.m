function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
% cost = 0; -----------------
% W1grad = zeros(size(W1)); -----------------
% W2grad = zeros(size(W2)); -----------------
% b1grad = zeros(size(b1)); -----------------
% b2grad = zeros(size(b2)); -----------------

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

[m n] = size(data);
%=======================================================================
% Forward pass:
inta1 = W1*data + repmat(b1,1,n);
inta2 = 1./(1+exp(-inta1));
clear inta1;
inta3 = W2*inta2 + repmat(b2,1,n);
output_sm = 1./(1+exp(-inta3));
clear inta3;
%== First term of cost function: fidelity term
vect_term_1 = ((output_sm - data).^2)*0.5;
cost_term_1 = sum(vect_term_1(:))/n;
clear vect_term_1;
%== Second term of cost function: weight decay
cost_term_2 = (lambda/2)*(sum(W1(:).^2) + sum(W2(:).^2));
%== Third term of cost function: sparsity penalty term
pj_estimated = sum(inta2,2)/n;
% pj_estimated_wv = (1/n)*inta2;
vect_term_3 = (sparsityParam*(log(sparsityParam./pj_estimated)))...
    +((1-sparsityParam)*(log((1-sparsityParam)./(1-pj_estimated))));
cost_term_3 = beta*sum(vect_term_3(:));
clear vect_term_3;

cost = cost_term_1 + cost_term_2 + cost_term_3;
%=======================================================================
% Backpropagation:
dC_din2 = ((1/n)*(output_sm - data)).*(output_sm.*(1-output_sm));
clear output_sm;
% auxxx = ((1/n)*(output_sm - data));
% dC_din2 = auxxx.*(output_sm.*(1-output_sm));
% W2grad = dC_din2*inta2'; ONE
W2grad = (dC_din2*inta2') + (lambda*W2);
% b2grad = dC_din2*ones(size(inta2')); MAL
[m1 n1] = size(inta2);
b2grad = dC_din2*ones(n1,1);
% dC_din1 = (W2'*dC_din2).*inta2.*(1-inta2); TWO
% dC_din1 = (W2'*dC_din2 + beta*(-(sparsityParam./pj_estimated_wv) +...
%     ((1-sparsityParam)./(1-pj_estimated_wv)) )).*inta2.*(1-inta2); MAL

% dC_din1 = (W2'*dC_din2).*inta2.*(1-inta2);
% dC_din12 = (beta*repmat((((1-sparsityParam)./(1-pj_estimated))-(sparsityParam./pj_estimated)),1,n)).*inta2.*(1-inta2);
vect_term = (beta/n)*(((1.-sparsityParam)./(1.-pj_estimated))-(sparsityParam./pj_estimated));
dC_din1 = (W2'*dC_din2 + repmat(vect_term,1,n)).*inta2.*(1.-inta2);
clear dC_din2;
clear W2;
% W1grad = dC_din1*data'; ONE
% W1grad = (dC_din1*data') + (lambda*W1); TWO
% W1grad = (dC_din1*data') + (lambda*W1) + repmat(dC_din12,1,n1)*data';

% W1grad = (dC_din1*data') + (lambda*W1) + dC_din12*data';
W1grad = (dC_din1*data') + (lambda*W1);
clear W1;
% b1grad = dC_din1*ones(size(data')); MAL
b1grad = dC_din1*ones(n,1);
%===================














%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.
% size(W1grad)
% size(W2grad)
% size(b1grad)
% size(b2grad)
grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
