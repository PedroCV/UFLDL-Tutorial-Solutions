function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);

pooledFeatures = zeros(numFeatures, numImages, floor(convolvedDim / poolDim), floor(convolvedDim / poolDim));

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------
dim_h = floor(convolvedDim / poolDim);

% for i = 1:dim_h
%     for j = 1:dim_h
%         
%         
%         
%         pooledFeatures = zeros(numFeatures, numImages, floor(convolvedDim / poolDim), floor(convolvedDim / poolDim));
%     end
% end

q_tot = poolDim^2;
avg_kern = ones(poolDim)/q_tot;
% div_sample = floor(convolvedDim / poolDim);
% current_image = zeros(size(convolvedDim,3),size(convolvedDim,3));
for imageNum = 1:numImages
  for featureNum = 1:numFeatures
%     Convolution of the current image with average kernel:
    current_image = squeeze(convolvedFeatures(featureNum,imageNum,:,:));
    Img_conv = conv2(current_image,avg_kern,'valid');
%     Downsampling to get the correct
    aux = downsample(Img_conv,poolDim);
    aux1 = downsample(aux',poolDim);
    aux1 = aux1';
%     
    pooledFeatures(featureNum,imageNum,:,:) = aux1;
  end
end


end
