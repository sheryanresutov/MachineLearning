function [ ] = binGaussianGenerativeModel(x, y, x1, mu, s, phi)
    %x - features (2 columns)
    %y - labels (2 classes)
    %x1 - the support of feature 1
    %mu - 1x2 matrix
    %s - arbitrary constant
    %phi - basis function taking in x and mean
    
    features = x;
    featuresTest = x;
    labels = y;
    labelsTest = y;

    phiX1 = phi(x1, mu(1));
    featuresTest(:,1) = phi(featuresTest(:,1), mu(1));
    featuresTest(:,2) = phi(featuresTest(:,2), mu(2));
    class0 = phi(features(labels == 0,:), mu(1));
    class1 = phi(features(labels == 1,:), mu(2));

    invCov = inv(cov(features));
    mu1 = mean(class0)';
    mu2 = mean(class1)';

    w = invCov*(mu1 - mu2);
    w0 = -.5*mu1'*invCov*mu1 + .5*(mu2'*invCov*mu2) + log(length(class0)/length(class1));

    phiX2 = (-w0 - w(1)*phiX1)/w(2); % w0 + w1(x1) + w2(x2) = 0, fe

    guess = w'*featuresTest' + w0 < 0; % if < 0 logistic, sigmoid > 0.5
    accuracy = sum(guess'==labelsTest)/length(guess);

    figure;
    hold on;
    scatter(class0(:,1), class0(:,2), 'r');
    scatter(class1(:,1), class1(:,2), 'b');
    plot(phiX1, phiX2, 'g');
    legend('class0', 'class1', 'line');
    hold off;
end
