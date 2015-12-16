% Sheryan Resutov, Eugene Sokolov, Harrison Zhao
% Machine Learning extra credit
clc; clear all; close all;
K = 4;
Xs = cell(1,4);
Xs{1} = mvnrnd([1 1], 0.025*eye(2), 150);
Xs{2} = mvnrnd([0 0], diag([0.05 0.05]), 200);
Xs{3} = mvnrnd([0 1], diag([0.01 0.075]), 250);
Xs{4} = mvnrnd([1 0], 0.025*eye(2), 300);

muOld = {[2 2], [-1 -1], [-1, 2], [2, -1]};
sigOld = cell(1,4);
pOld = cell(1,4);
for ii=1:K
    sigOld{ii} = 0.25*eye(2);
    pOld{ii} = 0.1; 
end
X = [];
for ii=1:K
    X = [X; Xs{ii}];
end
[N, D] = size(X);

err = 1;
ll = 1;
threshold = 1e-2;
iter = 1;
t = linspace(0,2*pi,1000);
while (err(iter) > threshold)
    iter = iter + 1;
    denominator = 0;
    gamma = zeros(N,K);
    for ii=1:K
        gamma(:,ii) = pOld{ii}*mvnpdf(X,muOld{ii},sigOld{ii});
        denominator = denominator + gamma(:,ii);
    end
    for ii=1:K
        gamma(:,ii) = gamma(:,ii)./denominator;
    end
    Nk = sum(gamma,1);
    muNew = (X'*gamma)./repmat(Nk,D,1);
    sigNew = cell(1,K);
    for ii=1:K
        for jj=1:N
            diff = X(jj,:)'-muNew(:,ii);
            sigNew{ii}(:,:,jj) = gamma(jj,ii)*diff*(diff)'/Nk(ii);
        end
        sigNew{ii} = sum(sigNew{ii},3);
    end
    pNew = Nk/N;
    tmpLL = [];
    for ii=1:K
        tmpLL = [tmpLL, pNew(ii)*mvnpdf(X,muNew(:,ii)',sigNew{ii})];
    end
    ll(iter) = sum(log(sum(tmpLL,2)));
    err(iter) = abs(ll(iter) - ll(iter-1));
    for ii=1:K
        pOld{ii} = pNew(ii);
        muOld{ii} = muNew(:,ii)';
        sigOld{ii} = sigNew{ii};
    end
    
    a = cell(1,K);
    b = cell(1,K);
    x = cell(1,K);
    y = cell(1,K);
    for ii=1:K
        [r c] = eig(sigNew{ii});
        angle = atan(r(1,2)/r(1,1));
        a{ii} = sqrt(c(1,1));
        b{ii} = sqrt(c(2,2));
        x{ii} = a{ii}*sin(t+angle)+muNew(1,ii);
        y{ii} = b{ii}*cos(t)+muNew(2,ii);
    end
    
    figure;
    hold on;
    plot(X(:,1),X(:,2),'r.');
    plot(muNew(1,:),muNew(2,:),'gX','Markersize',16,'LineWidth',8);
    for ii=1:K
        plot(x{ii},y{ii},'g.');
    end
    title(sprintf('cluster centroids with iterations=%d and err=%f', iter, err(iter)));
end


