function [ ] = logRegression(x, y, x1, mu, s, phi)
    %x - features (2 columns)
    %y - labels (2 classes)
    %x1 - the support of feature 1
    %mu - 1x2 matrix
    %s - arbitrary constant
    %phi - basis function
    
    features = x;
    featuresTest = x;
    labels = y;
    labelsTest = y;
    
    designMatrix = [ones(length(labels), 1) phi(features(:,1), mu(1)) phi(features(:,2), mu(2))];
    logSigmoid = @(a) 1/(1+exp(-a));
    w = rand(3,1);
    gety = @(w, x) (w'*[ones(1, size(x,1)); x'])';
    getR = @(y) diag(y .* (1- y));
    err = 1;
    while (err >= 1e-2)
        y = gety(w,features);
        R = getR(y);
        error = y - labels;
        z = designMatrix*w - inv(R)*(error);
        w = inv(designMatrix'*R*designMatrix)*designMatrix'*R*z;
        err = sum(error .^ 2)/length(labels)
    end
    
end

