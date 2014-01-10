function [cost, grad] = sparseCodingFeatureCost(weightMatrix, featureMatrix, visibleSize, numFeatures, patches, gamma, lambda, epsilon, groupMatrix)
%sparseCodingFeatureCost - given the weights in weightMatrix,
%                          computes the cost and gradient with respect to
%                          the features, given in featureMatrix
% parameters
%   weightMatrix  - the weight matrix. weightMatrix(:, c) is the cth basis
%                   vector.
%   featureMatrix - the feature matrix. featureMatrix(:, c) is the features
%                   for the cth example
%   visibleSize   - number of pixels in the patches
%   numFeatures   - number of features
%   patches       - patches
%   gamma         - weight decay parameter (on weightMatrix)
%   lambda        - L1 sparsity weight (on featureMatrix)
%   epsilon       - L1 sparsity epsilon
%   groupMatrix   - the grouping matrix. groupMatrix(r, :) indicates the
%                   features included in the rth group. groupMatrix(r, c)
%                   is 1 if the cth feature is in the rth group and 0
%                   otherwise.

    if exist('groupMatrix', 'var')
        assert(size(groupMatrix, 2) == numFeatures, 'groupMatrix has bad dimension');
    else
        groupMatrix = eye(numFeatures);
    end

    numExamples = size(patches, 2);

    weightMatrix = reshape(weightMatrix, visibleSize, numFeatures);
    featureMatrix = reshape(featureMatrix, numFeatures, numExamples);

    % -------------------- YOUR CODE HERE --------------------
    % Instructions:
    %   Write code to compute the cost and gradient with respect to the
    %   features given in featureMatrix.     
    %   You may wish to write the non-topographic version, ignoring
    %   the grouping matrix groupMatrix first, and extend the 
    %   non-topographic version to the topographic version later.
    % -------------------- YOUR CODE HERE --------------------
    %===========
    %Computing COST:
    term_dif = weightMatrix*featureMatrix - patches;
    aux1 = term_dif.^2;
    cost_term1 = sum(aux1(:))./numExamples;
    
%     aux = groupMatrix*(featureMatrix*featureMatrix');
%     aux1 = sqrt(sum(aux,2) + epsilon);

%     aux2 = epsilon + groupMatrix*(featureMatrix*featureMatrix');
    aux2 = epsilon + groupMatrix*(featureMatrix.^2);
    aux3 = sqrt(aux2);
    cost_term2 = (lambda/numExamples).*sum(aux3(:));
    
% % %== DOS
% %     aux = groupMatrix*(featureMatrix*featureMatrix');
% %     aux1 = sqrt(sum(aux,2) + epsilon);
%     aux2 = abs(groupMatrix*featureMatrix);
% %     aux2 = epsilon + groupMatrix*(featureMatrix.^2);
%     cost_term2 = lambda*sum(aux2(:));
% % %==
    
    aux = weightMatrix.^2;
    cost_term3 = gamma.*sum(aux(:));
    
    cost = cost_term1 + cost_term2 + cost_term3;
    %===========
    %Computing Gradient:
    aux4 = (2/numExamples).*((weightMatrix')*term_dif);
    
% % %     aux2 = (epsilon + groupMatrix*(featureMatrix*featureMatrix')).^(-1/2);
% % %     aux5 = (epsilon + (featureMatrix.^2)).^(-1/2);
% %     aux5 = featureMatrix.*(1./sqrt(epsilon + (featureMatrix.^2)));
% % %     grad = aux4 + lambda.*(featureMatrix.*aux5);
% %     aux6 = aux4 + (lambda.*aux5);
% %     
    
%     aux2 = (epsilon + groupMatrix*(featureMatrix*featureMatrix')).^(-1/2);
%     aux5 = (epsilon + (featureMatrix.^2)).^(-1/2);

%     aux5 = (groupMatrix')*(1./sqrt(epsilon + (groupMatrix*(featureMatrix*featureMatrix'))))*featureMatrix;
    
aux5 = (groupMatrix')*(1./aux3).*featureMatrix;

%     aux5 = (groupMatrix*featureMatrix)./sqrt(epsilon + (groupMatrix*(featureMatrix.^2)));
%     aux5 = ((groupMatrix')./sqrt(epsilon + (groupMatrix*(featureMatrix*featureMatrix')))).*featureMatrix;
    %     grad = aux4 + lambda.*(featureMatrix.*aux5);
    aux6 = aux4 + ((lambda/numExamples).*aux5);
    grad = aux6(:);
end
