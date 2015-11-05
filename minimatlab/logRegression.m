function [ ] = logRegression(x, y, x1, mu, s, phi)
    %x - features (2 columns)
    %y - labels (2 classes)
    %x1 - the support of feature 1
    %mu - 1x2 matrix
    %s - arbitrary constant
    %phi - basis function
    
    features = x;
    labels = y;
    
    features = [phi(features(:,1), mu(1)) phi(features(:,2), mu(2))];
    w = rand(3,1);
    designMatrix = [ones(length(labels),1) features];
    
    logSigmoid = @(a) 1./(1+exp(-a)); 
    err = 1;
    while(err >= 1e-3)
        prevErr = err;
        y = logSigmoid(w' * designMatrix')';
        R = diag(y .* (1 - y));
        error = y - labels;
        z = designMatrix*w - inv(R)*(error);
        w = inv(designMatrix'*R*designMatrix)*designMatrix'*R*z;
        err = sum(error .^ 2)/length(labels);
        if (prevErr == err)
            break;
        end
    end
    
    class0 = phi(x(labels == 0,:),mu(1));
    class1 = phi(x(labels == 1,:),mu(2));
    phiX1 = phi(x1, mu(1));
    phiX2 = (-w(1) - w(2)*phiX1)/w(3);
    % w0 + w1(x1) + w2(x2) = 0, fe
    
    figure;
    hold on;
    scatter(class0(:,1), class0(:,2), 'r');
    scatter(class1(:,1), class1(:,2), 'b');
    plot(phiX1, phiX2, 'g');
    legend('class0', 'class1', 'line');
    hold off;
end
