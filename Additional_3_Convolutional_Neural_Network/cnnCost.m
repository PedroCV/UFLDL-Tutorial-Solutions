function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                filterDim,numFilters,poolDim,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x numImges
%                array
%  numClasses -  number of classes to predict
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  poolDim    -  dimension of pooling area
%  pred       -  boolean only forward propagate and return
%                predictions
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)


if ~exist('pred','var')
    pred = false;
end;


imageDim = size(images,1); % height/width of image
numImages = size(images,3); % number of images

%% Reshape parameters and setup gradient matrices

% Wc is filterDim x filterDim x numFilters parameter matrix
% bc is the corresponding bias

% Wd is numClasses x hiddenSize parameter matrix where hiddenSize
% is the number of output units from the convolutional layer
% bd is corresponding bias
[Wc, Wd, bc, bd] = cnnParamsToStack(theta,imageDim,filterDim,numFilters,...
                        poolDim,numClasses);

% Same sizes as Wc,Wd,bc,bd. Used to hold gradient w.r.t above params.
Wc_grad = zeros(size(Wc));
Wd_grad = zeros(size(Wd));
bc_grad = zeros(size(bc));
bd_grad = zeros(size(bd));

%%======================================================================
%% STEP 1a: Forward Propagation
%  In this step you will forward propagate the input through the
%  convolutional and subsampling (mean pooling) layers.  You will then use
%  the responses from the convolution and pooling layer as the input to a
%  standard softmax layer.

%% Convolutional Layer
%  For each image and each filter, convolve the image with the filter, add
%  the bias and apply the sigmoid nonlinearity.  Then subsample the 
%  convolved activations with mean pooling.  Store the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled.  You will need to save the convolved activations for
%  backpropagation.
convDim = imageDim-filterDim+1; % dimension of convolved output
outputDim = (convDim)/poolDim; % dimension of subsampled output

% convDim x convDim x numFilters x numImages tensor for storing activations
activations = zeros(convDim,convDim,numFilters,numImages);

% outputDim x outputDim x numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%% YOUR CODE HERE %%%
activations = cnnConvolve_1(filterDim, numFilters, images, Wc, bc);
activationsPooled = cnnPool_1(poolDim, activations);

% Reshape activations into 2-d matrix, hiddenSize x numImages,
% for Softmax layer
activationsPooled = reshape(activationsPooled,[],numImages);

%% Softmax Layer
%  Forward propagate the pooled activations calculated above into a
%  standard softmax layer. For your convenience we have reshaped
%  activationPooled into a hiddenSize x numImages matrix.  Store the
%  results in probs.

% numClasses x numImages for storing probability that each image belongs to
% each class.
probs = zeros(numClasses,numImages);
M = size(images, 3);
%%% YOUR CODE HERE %%%
aux1 = Wd*activationsPooled + repmat(bd,1,M);
aux2 = bsxfun(@minus, aux1, max(aux1, [], 1)); %Substracts the maximm value of the matrix "aux1".
aux3 = exp(aux2);
probs = bsxfun(@rdivide, aux3, sum(aux3)); %It divides the vector "aux3" by the sum of its elements.
% clear aux1;
clear aux2;
clear aux3;
%%======================================================================
%% STEP 1b: Calculate Cost
%  In this step you will use the labels given as input and the probs
%  calculate above to evaluate the cross entropy objective.  Store your
%  results in cost.
groundTruth = full(sparse(labels, 1:M, 1));
aux4 = groundTruth.*probs;
aux5 = log(aux4(aux4 ~= 0)); %Extract non-zero entries.
%== Obly term of cost function: fidelity term
cost = -mean(aux5);
clear aux4;
clear aux5;
%%% YOUR CODE HERE %%%

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
%     preds = probs
    return;
end;

%%======================================================================
%% STEP 1c: Backpropagation
%  Backpropagate errors through the softmax and convolutional/subsampling
%  layers.  Store the errors for the next step to calculate the gradient.
%  Backpropagating the error w.r.t the softmax layer is as usual.  To
%  backpropagate through the pooling layer, you will need to upsample the
%  error with respect to the pooling layer for each filter and each image.  
%  Use the kron function and a matrix of ones to do this upsampling 
%  quickly.

%%% YOUR CODE HERE %%%
deriv_1 = (-1/M).*(groundTruth - probs);
clear groundTruth;
% Wd_grad = (activationsPooled*deriv_1')';
Wd_grad = deriv_1*activationsPooled';
clear activationsPooled;
bd_grad = deriv_1*ones(M,1);
deriv_2_pooled_sh = Wd'*deriv_1;
clear deriv_1;
deriv_2_pooled = reshape(deriv_2_pooled_sh,outputDim,outputDim,numFilters,numImages);
deriv_2_upsampled = zeros(convDim,convDim,numFilters,numImages);
for imageNum = 1:numImages
  im = squeeze(images(:,:,imageNum));
  for filterNum = 1:numFilters
%==   Upsampling:
%     aux2 = upsample(deriv_2_pooled(:,:,filterNum,imageNum),poolDim);
%     aux3 = upsample(aux2',poolDim);
%     deriv_2_upsampled(:,:,filterNum,imageNum) = (aux3').*activations(:,:,filterNum,imageNum).*(1-activations(:,:,filterNum,imageNum));
%   Upsample the incoming error using kron
    aux3 = (1/(poolDim^2)).*kron(squeeze(deriv_2_pooled(:,:,filterNum,imageNum)),ones(poolDim));
    deriv_2_upsampled(:,:,filterNum,imageNum) = aux3.*activations(:,:,filterNum,imageNum).*(1-activations(:,:,filterNum,imageNum));
%==   Convolution:
%     im = squeeze(images(:,:,imageNum));
    f_now = squeeze(deriv_2_upsampled(:,:,filterNum,imageNum));
    noww = conv2(im,rot90(squeeze(f_now),2),'valid');
%     size(noww)
%     size(Wc_grad(:,:,filterNum))
    Wc_grad(:,:,filterNum) = squeeze(Wc_grad(:,:,filterNum)) + noww; 
    bc_grad(filterNum) = bc_grad(filterNum) + sum(f_now(:));
%     activations = zeros(convDim,convDim,numFilters,numImages);
    
% %   Convolution of the current image with average kernel:
%     current_image = squeeze(convolvedFeatures(:,:,filterNum,imageNum));
%     Img_conv = conv2(current_image,avg_kern,'valid');
% %   Upsampling to get the correct
%     aux = downsample(Img_conv,poolDim);
%     aux1 = downsample(aux',poolDim);
%     aux1 = aux1';
% %   
%     pooledFeatures(:,:,filterNum,imageNum) = aux1;
  end
end
%  Wc      -  filterDim x filterDim x numFilters parameter matrix
%  Wd      -  numClasses x hiddenSize parameter matrix, hiddenSize is
%             calculated as numFilters*((imageDim-filterDim+1)/poolDim)^2 
%  bc      -  bias for convolution layer of size numFilters x 1
%  bd      -  bias for dense layer of size hiddenSize x 1
%
%
% aux2 = upsample(deriv_2_pooled,poolDim);
% aux3 = upsample(aux2',poolDim);
% deriv_2 = (aux3').*activations.*(1-activations);
%
% activations = zeros(convDim,convDim,numFilters,numImages);
% activationsPooled = zeros(outputDim,outputDim,numFilters,numImages);

%%======================================================================
%% STEP 1d: Gradient Calculation
%  After backpropagating the errors above, we can use them to calculate the
%  gradient with respect to all the parameters.  The gradient w.r.t the
%  softmax layer is calculated as usual.  To calculate the gradient w.r.t.
%  a filter in the convolutional layer, convolve the backpropagated error
%  for that filter with each image and aggregate over images.

%%% YOUR CODE HERE %%%
clear activations;
clear deriv_2_pooled;
clear deriv_2_upsampled;


%% Unroll gradient into grad vector for minFunc
grad = [Wc_grad(:) ; Wd_grad(:) ; bc_grad(:) ; bd_grad(:)];

end
