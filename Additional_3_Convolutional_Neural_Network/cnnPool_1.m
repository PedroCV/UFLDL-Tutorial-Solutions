function pooledFeatures = cnnPool_1(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
%   matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 
%   
%   Use mean pooling here.

%%% YOUR CODE HERE %%%
q_tot = poolDim^2;
avg_kern = ones(poolDim)/q_tot;
% div_sample = floor(convolvedDim / poolDim);
% current_image = zeros(size(convolvedDim,3),size(convolvedDim,3));
for imageNum = 1:numImages
  for filterNum = 1:numFilters
%     Convolution of the current image with average kernel:
    current_image = squeeze(convolvedFeatures(:,:,filterNum,imageNum));
    Img_conv = conv2(current_image,avg_kern,'valid');
%     Downsampling to get the correct
    aux = downsample(Img_conv,poolDim);
    aux1 = downsample(aux',poolDim);
    aux1 = aux1';
%     
    pooledFeatures(:,:,filterNum,imageNum) = aux1;
  end
end

end

